from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention

def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def window_attention_forward(self, x, mask = None):


    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(name, Pretrained=True):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.

    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384

    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    net = timm.create_model(name, pretrained=Pretrained)
    net.features = {}

    def get_features_hook(layer_name):
        def hook(module, input, output):
            # 这里的 output 是我们需要的特征
            net.features['norm'] = output # 统一存入 'norm' 键，方便后续调用
        return hook

    # --- 改进的 Hook 注册逻辑 ---
    target_layer = None

    # 遍历所有层，寻找最后一个出现的 'norm' 层
    # 对于 ViT 它是 net.norm, 对于 Swin 它是 net.layers.3.norm
    for layer_name, module in net.named_modules():
        # 1. 记录层名中包含 'norm' 的最后一个模块
        if 'norm' in layer_name.lower():
            target_layer = (layer_name, module)

    if target_layer is not None:
        ln, mod = target_layer
        mod.register_forward_hook(get_features_hook(ln))
        print(f"✅ Successfully hooked: {ln}")
    else:
        # 最后的兜底：如果完全没 norm，尝试 Hook 最后一个 Block
        print("⚠️ Warning: No 'norm' found, hooking the last module instead.")
        last_mod_name, last_mod = list(net.named_modules())[-1]
        last_mod.register_forward_hook(get_features_hook(last_mod_name))

    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)

    net = net.cuda()
    net.eval()
    return net
