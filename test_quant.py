
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import argparse
import random

import IPython
import torch

from utils import *
from quant import *
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.build_model import build_model
from utils.data_utils import build_dataset



class SelfDefinedDataset(Dataset):
    def __init__(self, pil_images, transform=None):
        self.pil_images = pil_images
        self.transform = transform

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        image = self.pil_images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def img_reverse(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[c] = image_tensor[c] * s + m
    image_tensor = image_tensor * 255
    return image_tensor

def get_args_parser():
    parser = argparse.ArgumentParser(description="I&S-ViT", add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_tiny', 'vit_small', 'vit_base',
                                 'deit_tiny', 'deit_small', 'deit_base',
                                 'swin_tiny', 'swin_small', 'swin_base'],
                        help="model")
    parser.add_argument('--dataset', default="E:/data/ImageNet2012",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=128,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--val-batchsize", default=128,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=0, type=int,
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=200,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument('--w_bits', default=4,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4,
                        type=int, help='bit-precision of activation')
    parser.add_argument('--iters', default=100, type=int, help='number of iteration for optimization')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # parser.add_argument('--save_fake', action='store_true', help='save fake data')
    parser.add_argument('--rep', action="store_true", help='using repq')
    parser.add_argument("--fake_path", default=None, type=str, help="path for fake data")
    parser.add_argument("--box_path", default=None, type=str)
    parser.add_argument("--softtagets_path", default=None, type=str)
    parser.add_argument("--use_real_data", action='store_true')
    parser.add_argument("--use_Gaussian", action='store_true')

    return parser


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def usm(tensor, amount=2.0):
    """方法一：非锐化掩模 (USM)"""
    sigma = 1.0
    kernel_size = 3
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = (gauss.view(1, -1) * gauss.view(-1, 1)).expand(tensor.size(1), 1, kernel_size, kernel_size).to(tensor.device)
    low_freq = F.conv2d(tensor, kernel_2d, groups=tensor.size(1), padding=kernel_size//2)
    return torch.clamp(tensor + amount * (tensor - low_freq), 0, 1)


def fft(tensor, radius=10, strength=10):
    """方法二：频域高通滤波"""
    B, C, H, W = tensor.shape
    freq = torch.fft.fftn(tensor, dim=(-2, -1))
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((x - W//2)**2 + (y - H//2)**2).to(tensor.device)
    mask = torch.ones_like(dist)
    mask[dist > radius] += strength
    enhanced = torch.fft.ifftn(torch.fft.ifftshift(freq_shifted * mask.view(1, 1, H, W), dim=(-2, -1)), dim=(-2, -1))
    return torch.real(enhanced).clamp(0, 1)


def laplacian(tensor, strength=10):
    """方法三：拉普拉斯算子"""
    kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(tensor.device)
    edge = F.conv2d(tensor, kernel, groups=3, padding=1)
    return torch.clamp(tensor + strength * edge, 0, 1)

def wavelet(tensor, multiplier=10):
    """方法四：小波子带增强"""
    x00, x10, x01, x11 = tensor[:,:,0::2,0::2], tensor[:,:,1::2,0::2], tensor[:,:,0::2,1::2], tensor[:,:,1::2,1::2]
    ll, hl, lh, hh = (x00+x10+x01+x11)/4, (x00-x10+x01-x11)/4, (x00+x10-x01-x11)/4, (x00-x10-x01+x11)/4
    hl, lh, hh = hl*multiplier, lh*multiplier, hh*multiplier
    res = torch.zeros_like(tensor)
    res[:,:,0::2,0::2], res[:,:,1::2,0::2] = ll+hl+lh+hh, ll-hl+lh-hh
    res[:,:,0::2,1::2], res[:,:,1::2,1::2] = ll+hl-lh-hh, ll-hl-lh+hh
    return torch.clamp(res, 0, 1)

def main():
    print(args)
    seed(args.seed)

    model_zoo = {
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'vit_base': 'vit_base_patch16_224',

        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base': 'deit_base_patch16_224',

        'swin_tiny': 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base': 'swin_base_patch4_window7_224',
    }

    device = torch.device(args.device)

    # Prepare data
    # print('Building dataloader ...')
    train_loader, val_loader = build_dataset(args)


    if args.use_real_data:
        for data, _ in train_loader:
            calib_data = data.to(device)
            break
        calib_data.to(device)
    else:

        if args.fake_path is not None:
            # load fake data
            print(f"load fake data from {args.fake_path}")
            print('load fake data from', os.path.join('fake_data', f'{args.model}', args.fake_path))
            with open(os.path.join('fake_data', f'{args.model}', args.fake_path), "rb") as fp:  # Pickling
                calibrate_data = pickle.load(fp)
                # calibrate_data = train_transform(torch.from_numpy(calibrate_data)).to(device)
                calib_data = torch.from_numpy(calibrate_data).to(device)

        if args.box_path is not None:
            print('load fake box from', os.path.join('fake_data', f'{args.model}', args.box_path))
            with open(os.path.join('fake_data', f'{args.model}', args.box_path), "rb") as fp:  # Pickling
                boxes = pickle.load(fp)
            resize_transform = transforms.Resize((224,224))
            img_cropped = []
            img_cropped_label = []
            for i in range(len(boxes)):
                if len(boxes[i]) == 1:
                    continue
                for j in range(len(boxes[i])):
                    current = calib_data[i]
                    box = boxes[i][j]
                    # cropped_label = softtagets[i][j]
                    img_cropped.append(resize_transform(transforms.functional.crop(current, top=box[0], left=box[1],
                                                                                   height=box[2], width=box[3])))

            img_cropped = torch.stack(img_cropped).cuda()
            calib_data = torch.cat((calib_data, img_cropped))

    if args.use_Gaussian:
        print('use_Gaussian!')
        calib_data = torch.randn((32,3,224,224))
        calib_data = calib_data.to(device)

    # calib_data = usm(calib_data)
    # calib_data = fft(calib_data)
    # calib_data = laplacian(calib_data)
    # calib_data = wavelet(calib_data)
    print('calib_data.shape', calib_data.shape)
    # Prepare model
    print('Building model ...')
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()

    import copy
    fp_model = copy.deepcopy(model)

    # Quantization setting
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    print("quantization settings:", wq_params, "|", aq_params)
    print()

    # Wrap quantized model
    if args.rep:
        q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params, rep=args.rep)
    else:
        q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    # Initial quantizations
    print("Stage One, Q-Act, FP-W")
    set_quant_state(q_model, input_quant=True, weight_quant=False)

    # Obtain quantization parameters for act
    print("Init quantization parameters of act")
    with torch.no_grad():
        _ = q_model(calib_data[:32])

    def recon_model(model: nn.Module, teacher_model: nn.Module):
        """
        Reconstruction for blocks and layers which assigned
        """
        for a, b in zip(model.named_children(), teacher_model.named_children()):
            name, module = a
            _, T_module = b
            if isinstance(module, (QuantConv2d, QuantLinear)):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(q_model, module, fp_model, T_module, **kwargs)
            elif isinstance(module, type(q_model.blocks[0])):
                print('Reconstruction for block {}'.format(name))
                block_reconstruction(q_model, module, fp_model, T_module, **kwargs)
            else:
                recon_model(module, T_module)

    def recon_model_swin(model: nn.Module, teacher_model: nn.Module):
        for a, b in zip(model.named_children(), teacher_model.named_children()):
            name, module = a
            _, T_module = b
            if isinstance(module, (QuantConv2d, QuantLinear)):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(q_model, module, fp_model, T_module, **kwargs)
            elif isinstance(module, type(q_model.layers[0].blocks[0])):
                print('Reconstruction for block {}'.format(name))
                block_reconstruction(q_model, module, fp_model, T_module, **kwargs)
            else:
                recon_model_swin(module, T_module)

    kwargs = dict(cali_data=calib_data, asym=True,
                  warmup=args.warmup, act_quant=True, weight_quant=False, opt_mode='mse', batch_size=16,
                  iters=args.iters,
                  lr=4e-5 # 论文初始学习率 # 使用Adam优化器 # 权重衰减为
                  )
    # first and last layer require optimization
    q_model.patch_embed.proj.ignore_reconstruction = False
    q_model.head.ignore_reconstruction = False
    # for swin, the reduction module requires optimization
    if 'swin' in args.model:
        for n, m in q_model.named_modules():
            if 'reduction' in n:
                m.ignore_reconstruction = False
    print("Start optimization")
    if "swin" in args.model:
        recon_model_swin(q_model, fp_model)
    else:
        recon_model(q_model, fp_model)

    set_quant_state(q_model, input_quant=True, weight_quant=False)
    if args.rep:
        print("Stage Two, reparameterization")
        with torch.no_grad():
            module_dict = {}
            q_model_slice = q_model.layers if 'swin' in args.model else q_model.blocks
            for name, module in q_model_slice.named_modules():
                module_dict[name] = module
                idx = name.rfind('.')
                if idx == -1:
                    idx = 0
                father_name = name[:idx]
                if father_name in module_dict:
                    father_module = module_dict[father_name]
                else:
                    raise RuntimeError(f"father module {father_name} not found")

                if 'norm1' in name or 'norm2' in name:
                    if 'norm1' in name:
                        next_module = father_module.attn.qkv
                    elif 'norm2' in name:
                        next_module = father_module.mlp.fc1

                    act_delta = next_module.input_quantizer.delta.reshape(-1)
                    act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                    act_min = -act_zero_point * act_delta

                    target_delta = torch.mean(act_delta)
                    target_zero_point = torch.mean(act_zero_point)
                    target_min = -target_zero_point * target_delta

                    r = act_delta / target_delta
                    b = act_min / r - target_min

                    module.weight.data = module.weight.data / r
                    module.bias.data = module.bias.data / r - b

                    next_module.weight.data = next_module.weight.data * r
                    if next_module.bias is not None:
                        next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data,
                                                                                 b.reshape(-1, 1)).reshape(-1)

                    else:
                        next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                        next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1, 1)).reshape(-1)

                    next_module.input_quantizer.channel_wise = False
                    next_module.input_quantizer.delta = Parameter(target_delta).contiguous()
                    next_module.input_quantizer.zero_point = Parameter(target_zero_point).contiguous()
                    next_module.weight_quantizer.inited.fill_(0)
    else:
        print("No Stage Two, reparameterization")

    print("Stage Three, Q-Act, Q-W")
    print("Init quantization parameters of weight")

    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data[:32])


    kwargs = dict(cali_data=calib_data, asym=True,
                  warmup=args.warmup, act_quant=True, weight_quant=True, opt_mode='mse', batch_size=16, iters=args.iters,
                  last_stage=True)
    print("re optimization")
    if "swin" in args.model:
        recon_model_swin(q_model, fp_model)
    else:
        recon_model(q_model, fp_model)

    set_quant_state(q_model, input_quant=True, weight_quant=True)
    print("Acc after re optimization")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, q_model, criterion, device
    )
    print()


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()


    pre_correct_list = []
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        maxk = max((1,))
        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        pre_correct_list.extend(correct.squeeze_().cpu().tolist())

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))
    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('I&S-ViT', parents=[get_args_parser()])
    args = parser.parse_args()

    main()


