import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
from tqdm import tqdm
from scipy.stats import multivariate_normal

from utils.build_model import build_model
import IPython
import torch.fft as fft
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import torch
import os



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


def attnmap_targets(args):
    def single_rv():
        mean = np.random.uniform(-5, 5, 2)
        covariance = None
        while True:
            A = np.random.uniform(0, 1, (2, 2)) * 10
            covariance = np.dot(A, A.T)
            try:
                np.linalg.cholesky(covariance)
                break
            except np.linalg.LinAlgError:
                continue

        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvalues = np.clip(eigenvalues, 0, 5)
        covariance = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)

        # print(mean, covariance[0], covariance[1])
        return multivariate_normal(mean, covariance)

    def single_attnmap():
        # rvnum is a Gaussian number, which means how many attention centers are in attnmap, randomly between 1-5
        # rvnum_candi = [1,2,3,4,5]
        rvnum_candi = [1]
        rvnum = rvnum_candi[random.randint(0, len(rvnum_candi) - 1)]

        rv = []
        for i in range(rvnum):
            rv.append(single_rv())

        width_candi = [4, 6, 8]
        width = width_candi[random.randint(0, len(width_candi) - 1)]

        x = np.linspace(-width, width, 14)
        y = np.linspace(-width, width, 14)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        values = []
        for i in range(rvnum):
            values.append(rv[i].pdf(pos))
        values = np.array(values)

        values_mix = np.max(values, axis=0) + 1e-9  # 1e-9 is set to avoid division by 0
        values_mix = values_mix / np.max(values_mix)

        values_softmax = F.softmax(torch.from_numpy(values_mix).reshape(-1))
        values_softmax = values_softmax / values_softmax.max()
        if torch.isnan(values_softmax).sum() > 0:
            import IPython
            print("single_attnmap debug")
            IPython.embed()

        # print(pos.shape, values.shape)
        logging.info('generate single attnmap: rvnum={}, width={}'.format(rvnum, width))
        assert 196 == values_softmax.numel()
        return values_softmax, rvnum

    if args.model == 'deit_small':
        head_num = 6
    elif args.model == 'deit_tiny':
        head_num = 3
    elif args.model == 'deit_base':
        head_num = 12
    else:
        raise NotImplementedError

    targets = []
    rvnums = []
    # for _ in range(args.batch_size):
    for _ in range(args.batch_size * head_num):
        value_cls = torch.rand(1)  
        vaule_patch, rvnum = single_attnmap()  
        vaule_patch = (vaule_patch / vaule_patch.sum()) * (1 - value_cls)  

        # assert 1.0 == value_cls.item()+vaule_patch.sum().item()
        logging.info('rvnum={}, cls={}, patch={}, sum={}'.format(rvnum, value_cls.item(), vaule_patch.sum().item(),
                                                                 value_cls.item() + vaule_patch.sum().item()))
        if torch.isnan(vaule_patch).sum() > 0:
            import IPython
            print("vaule_patch debug")
            IPython.embed()
        targets.append(torch.cat([value_cls, vaule_patch]))
        rvnums.append(rvnum)

    return torch.cat(targets).reshape(args.batch_size, head_num, -1).float(), rvnums


def attnmap_targets_fined(args, batchsize=None):
    state = np.random.get_state()
    np.random.seed(42)

    def single_rv():
        mean = np.random.uniform(-5, 5, 2)
        covariance = None
        while True:
            A = np.random.uniform(0, 1, (2, 2)) * 10
            covariance = np.dot(A, A.T)
            try:
                eigenvalues, eigenvectors = np.linalg.eig(covariance)
                eigenvalues = np.clip(eigenvalues, 0, 5)
                covariance = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)

                # print(mean, covariance[0], covariance[1])
                return multivariate_normal(mean, covariance)
            except np.linalg.LinAlgError:
                continue

    def single_attnmap():
      
        rvnum_candi = [1, 2, 3, 4, 5]
        rvnum = rvnum_candi[np.random.randint(0, len(rvnum_candi) - 1)]

        rv = []
        for i in range(rvnum):
            rv.append(single_rv())

       
        width_candi = [4, 6, 8]
        width = width_candi[np.random.randint(0, len(width_candi) - 1)]

        x = np.linspace(-width, width, 14)
        y = np.linspace(-width, width, 14)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        values = []
        for i in range(rvnum):
            values.append(rv[i].pdf(pos))
        values = np.array(values)
        values_mix = np.max(values, axis=0) + 1e-9  # 1e-9 is set to avoid division by 0
        values_mix = values_mix / np.max(values_mix)

        values_softmax = F.softmax(torch.from_numpy(values_mix).reshape(-1))
        values_softmax = values_softmax / values_softmax.max()
        if torch.isnan(values_softmax).sum() > 0:
            import IPython
            print("single_attnmap debug")
            IPython.embed()

        # print(pos.shape, values.shape)
        logging.info('generate single attnmap: rvnum={}, width={}'.format(rvnum, width))
        assert 196 == values_softmax.numel()
        return values_softmax, rvnum

    def single_attnmap_swin():
      
        rvnum_candi = [1]  # [1, 2, 3, 4, 5]
        rvnum = rvnum_candi[np.random.randint(0, len(rvnum_candi))]

        rv = []
        for i in range(rvnum):
            rv.append(single_rv())

     
        width_candi = [1, 2, 3]
        width = width_candi[np.random.randint(0, len(width_candi) - 1)]

        x = np.linspace(-width, width, 7)
        y = np.linspace(-width, width, 7)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        values = []
        for i in range(rvnum):
            values.append(rv[i].pdf(pos))
        values = np.array(values)

     
        values_mix = np.max(values, axis=0) + 1e-9  # 1e-9 is set to avoid division by 0
        values_mix = values_mix / np.max(values_mix)

        values_softmax = F.softmax(torch.from_numpy(values_mix).reshape(-1))
        values_softmax = values_softmax / values_softmax.max()
        if torch.isnan(values_softmax).sum() > 0:
            import IPython
            print("single_attnmap debug")
            IPython.embed()

        # print(pos.shape, values.shape)
        logging.info('generate single attnmap: rvnum={}, width={}'.format(rvnum, width))
        assert 49 == values_softmax.numel()
        return values_softmax, rvnum

    if args.model == 'deit_small':
        head_num = 6
        layers = 12
    elif args.model == 'deit_tiny':
        head_num = 3
        layers = 12
    elif args.model == 'deit_base':
        head_num = 12
        layers = 12
    elif args.model == 'vit_small':
        head_num = 6
        layers = 12
    elif args.model == 'vit_base':
        head_num = 12
        layers = 12
    elif args.model == 'vit_tiny': 
        head_num = 3
        layers = 12
    elif args.model == 'swin_tiny':
        head_nums = [3, 6, 12, 24]
        layers = [2, 2, 6, 2]
    elif args.model == 'swin_small':
        head_nums = [3, 6, 12, 24]
        layers = [2, 2, 18, 2]
    elif args.model == 'swin_base':
        head_nums = [4, 8, 16, 32]
        layers = [2, 2, 18, 2]
    else:
        raise NotImplementedError

    targets = []
    rvnums = []
    if 'swin' in args.model:
        stage_attnpatches = [64, 16, 4, 1]
        for layer_index, layer_number in enumerate(layers):
            for sub_layer in range(layer_number):
                targets_tmp = []
                rvnums_tmp = []

                stage_attnpatch = stage_attnpatches[layer_index]
                head_num = head_nums[layer_index]

                for _ in range(batchsize * head_num * stage_attnpatch):
                    vaule_patch, rvnum = single_attnmap_swin()
                    # assert 1.0 == value_cls.item()+vaule_patch.sum().item()
                    vaule_patch = (vaule_patch / vaule_patch.sum())
                    logging.info('rvnum={}, patch={}'.format(rvnum, vaule_patch.sum().item()))
                    if torch.isnan(vaule_patch).sum() > 0:
                        import IPython
                        print("vaule_patch debug")
                        IPython.embed()
                    targets_tmp.append(vaule_patch)
                    rvnums_tmp.append(rvnum)
                # targets.append(torch.stack(targets_tmp).to('cuda').float())
                
                # rvnums.append(rvnums_tmp)
                if len(targets_tmp) > 0:
                    targets.append(torch.stack(targets_tmp).to('cuda').float())
                    rvnums.append(rvnums_tmp)
        return targets, rvnums
    else:
        for _ in range(batchsize * layers * head_num):
            value_cls = torch.rand(1)  # cls token的值
            vaule_patch, rvnum = single_attnmap()
            vaule_patch = (vaule_patch / vaule_patch.sum()) * (1 - value_cls)  # 保证cls token与其余token相加和为1

            # assert 1.0 == value_cls.item()+vaule_patch.sum().item()
            logging.info('rvnum={}, cls={}, patch={}, sum={}'.format(rvnum, value_cls.item(), vaule_patch.sum().item(),
                                                                     value_cls.item() + vaule_patch.sum().item()))
            if torch.isnan(vaule_patch).sum() > 0:
                import IPython
                print("vaule_patch debug")
                IPython.embed()
            targets.append(torch.cat([value_cls, vaule_patch]))
            rvnums.append(rvnum)

        np.random.set_state(state)
        return torch.cat(targets).reshape(batchsize, layers, head_num, -1).float(), rvnums

def fftmask(r1, r2, r3):
    # r1 is outside diameter of low-frequency
    # r2 is outside diameter of high-frequency
    # r3 is inside diameter of high-frequency
    S = 224
    R = S // 2
    R1 = R * r1
    R2 = R * r2
    R3 = R * r3
    lmask = torch.zeros((S, S))
    hmask = torch.zeros((S, S))
    for i in range(S):
        for j in range(S):
            dis = (i - (S - 1) / 2) ** 2 + (j - (S - 1) / 2) ** 2
            if (dis <= R1 ** 2):
                lmask[i, j] = 1
            if (dis <= R2 ** 2 and dis >= R3 ** 2):
                hmask[i, j] = 1
    lmask, hmask = lmask.cuda(), hmask.cuda()
    return lmask, hmask

def imgfft(calibrate_data, lmask, hmask):
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()
    denorm_data = (calibrate_data * std.expand_as(calibrate_data)) + mean.expand_as(calibrate_data)

    h, w = denorm_data.shape[-2], denorm_data.shape[-1]

    f = fft.fftn(denorm_data, dim=(2,3))
    f = torch.roll(f, (h // 2, w // 2), dims=(2, 3))
    data_l = f * lmask
    data_h = f * hmask
    data_l = torch.abs(fft.ifftn(data_l, dim=(2, 3)))
    data_l = (data_l - mean.expand_as(calibrate_data)) / std.expand_as(calibrate_data)
    data_h = torch.abs(fft.ifftn(data_h, dim=(2, 3)))
    data_h = (data_h - mean.expand_as(calibrate_data)) / std.expand_as(calibrate_data)
    return data_l, data_h


def get_val(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)

def generate_data_mulitTarget(args, pred=None, attnmap_pred=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    batch_size = len(pred)
    print('Building model ...')
    p_model = build_model(model_zoo[args.model])
    p_model = p_model.cuda()
    p_model.eval()

    # hook attnmap
    attnhooks = []

    # Init Gaussian noise
    img = torch.randn((batch_size, 3, 224, 224)).cuda()
    img.requires_grad = True
    SOFT_LABEL_CLASSES = 1000
    # delta = torch.zeros_like(img)

    # set soft labels
    if args.softlabel:
        # one-hot到soft: [B]->[B, 1000]
        softlabel = []
        softtagets = []
        softtagets_softlabel = []
        softboxes = []
        for i in range(len(pred)):
            label = torch.rand(SOFT_LABEL_CLASSES)  
            cls_nums = random.randint(1, 5)  
            cls_ids = [pred[i].cpu()]
            label[pred[i].cpu()] = random.randint(5, 10) 
            softtagets.append([])
            softtagets[-1].append(pred[i].cpu())

            while len(cls_ids) < cls_nums: 
                cls_id = torch.randint(0, SOFT_LABEL_CLASSES - 1, ())
                if cls_id not in cls_ids:
                    cls_ids.append(cls_id)
                    label[cls_id] = random.randint(5, 10)
                    softtagets[-1].append(cls_id)

            if len(softtagets[-1]) == 1:
                softboxes.append([])
                # height, width = np.random.randint(100, 200), np.random.randint(100, 200)
                # top, left = np.random.randint(0, 224 - height), np.random.randint(0, 224 - width)
                softboxes[-1].append((0, 0, 224, 224))
                softtagets_softlabel.append([])
            elif len(softtagets[-1]) == 2:
                softboxes.append([])
                # softboxes = [(0, 0, 224, 112), (0, 112, 224, 112)]
                height, width = np.random.randint(100, 200), np.random.randint(100, 112)
                top, left = np.random.randint(0, 224 - height), np.random.randint(0, 112 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(100, 200), np.random.randint(100, 112)
                top, left = np.random.randint(0, 224 - height), np.random.randint(112, 224 - width)
                softboxes[-1].append((top, left, height, width))

                softtagets_softlabel.append([])
                _, indices = torch.topk(label, 2)
                for indice in indices:
                    label_tmp = torch.rand(SOFT_LABEL_CLASSES)
                    label_tmp[indice] = random.randint(5, 10)
                    softtagets_softlabel[-1].append(torch.nn.functional.softmax(label_tmp, dim=0))

            elif len(softtagets[-1]) == 3:
                softboxes.append([])
                # softboxes = [(0, 0, 224, 74), (0, 74, 224, 148), (0, 74, 224, 222)]
                height, width = np.random.randint(100, 200), np.random.randint(70, 74)
                top, left = np.random.randint(0, 224 - height), np.random.randint(0, 74 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(100, 200), np.random.randint(70, 74)
                top, left = np.random.randint(0, 224 - height), np.random.randint(74, 148 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(100, 200), np.random.randint(70, 74)
                top, left = np.random.randint(0, 224 - height), np.random.randint(148, 222 - width)
                softboxes[-1].append((top, left, height, width))

                softtagets_softlabel.append([])
                _, indices = torch.topk(label, 3)
                for indice in indices:
                    label_tmp = torch.rand(SOFT_LABEL_CLASSES)
                    label_tmp[indice] = random.randint(5, 10)
                    softtagets_softlabel[-1].append(torch.nn.functional.softmax(label_tmp, dim=0))

            elif len(softtagets[-1]) == 4:
                softboxes.append([])
                # softboxes = [(0, 0, 112, 112), (0, 112, 112, 112), (112, 0, 112, 112), (112, 112, 112, 112),]
                height, width = np.random.randint(100, 112), np.random.randint(100, 112)
                top, left = np.random.randint(0, 112 - height), np.random.randint(0, 112 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(100, 112), np.random.randint(100, 112)
                top, left = np.random.randint(0, 112 - height), np.random.randint(112, 224 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(100, 112), np.random.randint(100, 112)
                top, left = np.random.randint(112, 224 - height), np.random.randint(0, 112 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(100, 112), np.random.randint(100, 112)
                top, left = np.random.randint(112, 224 - height), np.random.randint(112, 224 - width)
                softboxes[-1].append((top, left, height, width))

                softtagets_softlabel.append([])
                _, indices = torch.topk(label, 4)
                for indice in indices:
                    label_tmp = torch.rand(SOFT_LABEL_CLASSES)
                    label_tmp[indice] = random.randint(5, 10)
                    softtagets_softlabel[-1].append(torch.nn.functional.softmax(label_tmp, dim=0))

            elif len(softtagets[-1]) == 5:
                softboxes.append([])
                # softboxes = [(0, 0, 70, 70), (144, 0, 80, 80), (0, 144, 80, 80),
                # (144, 144, 80, 80), (70, 70, 74, 74)]
                height, width = np.random.randint(60, 70), np.random.randint(60, 70)
                top, left = np.random.randint(0, 70 - height), np.random.randint(0, 70 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(60, 70), np.random.randint(60, 70)
                top, left = np.random.randint(0, 70 - height), np.random.randint(224 - 70, 224 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(60, 70), np.random.randint(60, 70)
                top, left = np.random.randint(224 - 70, 224 - width), np.random.randint(0, 70 - height)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(60, 70), np.random.randint(60, 70)
                top, left = np.random.randint(224 - 70, 224 - width), np.random.randint(224 - 70, 224 - width)
                softboxes[-1].append((top, left, height, width))
                height, width = np.random.randint(60, 74), np.random.randint(60, 74)
                top, left = np.random.randint(70, 70 + 74 - width), np.random.randint(70, 70 + 74 - width)
                softboxes[-1].append((top, left, height, width))

                softtagets_softlabel.append([])
                _, indices = torch.topk(label, 5)
                for indice in indices:
                    label_tmp = torch.rand(SOFT_LABEL_CLASSES)
                    label_tmp[indice] = random.randint(5, 10)
                    softtagets_softlabel[-1].append(torch.nn.functional.softmax(label_tmp, dim=0))

            logging.info("cls ids is {}".format(cls_ids))
            softlabel.append(torch.nn.functional.softmax(label, dim=0))  # 最后过一个softmax保证和为1
        softlabel = torch.Tensor([t.numpy() for t in softlabel]).to('cuda')
        logging.info("soft label shape is {}".format(softlabel.shape))
        logging.info("soft label is {}".format(softlabel))

        resize_transform = transforms.Resize((224, 224))

    # Init optimizer
    # args.lr = 0.25 if 'swin' in args.model else 0.20
    args.lr = 0.1

    optimizer = optim.Adam([img], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
    # Set pseudo labels
  
    var_pred = random.uniform(2500 * batch_size / 32, 3000 * batch_size / 32)  # for batch_size 32

    if args.softlabel:
        cropped_number = 0
        for item in softtagets:
            if len(item) == 1:
                continue
            cropped_number += len(item)
    hooks = []
    for n, m in p_model.named_modules():
        if 'matmul2' in n:
            hooks.append(AttentionMap(m))

    if 'swin' in args.model:
        for n, m in p_model.named_modules():
            if 'attn_drop' in n:
                if args.softlabel:
                    attnhooks.append(AttentionMap(m, batch_size, cropped_number))
                else:
                    attnhooks.append(AttentionMap(m, batch_size))
    else:
        for m in p_model.blocks:
            if args.softlabel:
                attnhooks.append(AttentionMap(m.attn.attn_drop, batch_size, cropped_number))
            else:
                attnhooks.append(AttentionMap(m.attn.attn_drop, batch_size))

    if args.softlabel:
        var_pred_cropped = random.uniform(2500 * cropped_number / 32, 3000 * cropped_number / 32)

    # init attnmap targets
    if attnmap_pred is None:
        # attnmap target
        if 'swin' in args.model:
            attnmap_pred, rvnums = attnmap_targets_fined(args=args, batchsize=batch_size)
        else:
            attnmap_pred, rvnums = attnmap_targets_fined(args=args, batchsize=batch_size)
            attnmap_pred = attnmap_pred.to('cuda')
        if args.softlabel:
           
            if cropped_number > 0:
                if 'swin' in args.model:
                    attnmap_pred_cropped, rvnums_cropped = attnmap_targets_fined(args=args, batchsize=cropped_number)
                else:
                    attnmap_pred_cropped, rvnums_cropped = attnmap_targets_fined(args=args, batchsize=cropped_number)
                    attnmap_pred_cropped = attnmap_pred_cropped.to('cuda')
            else:
                attnmap_pred_cropped = None
                rvnums_cropped = []


    # set criterion
    criterion = nn.CrossEntropyLoss()
    KL_Loss = nn.KLDivLoss(reduction='batchmean')
    soft_criterion = CrossEntropyLossSoft(reduction='none')
    radius = [[0.3, 0.5, 0.2],[0.6, 1, 0]]
    loss_history = {
        'total_loss': [],
        'loss_oh': [],  #  (One-hot)
        'loss_soft': [], 
        'loss_tv': [], 
        'loss_attnmaps': [],  
        'loss_align': [],  # Frequency Alignment Total Loss (Align)
        'loss_kl': [],  # Frequency KL divergence
        'loss_hard': []  
    }
    # Train for two epochs

    for lr_it in range(2):
        # code from psaq
        if lr_it == 0:
            iterations_per_layer = 500
            lim = 15
        else:
            iterations_per_layer = 500
            lim = 30
        # iterations_per_layer = 200
        # lim = int(112 * 0.3 * (lr_it / 2))

        lr_scheduler = lr_cosine_policy(args.lr, 100, iterations_per_layer)
        # Filter out high and low frequency information
        lmask, hmask = fftmask(radius[lr_it][0], radius[lr_it][1], radius[lr_it][2])

        for itr in range(iterations_per_layer):
            # Learning rate scheduling
            lr_scheduler(optimizer, itr, itr)

            # psaq's code
            # Apply random jitter offsets (from DeepInversion[1])
            # [1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion.", CVPR2020.
            off = random.randint(-lim, lim)
            img_l, img_h = imgfft(img, lmask, hmask)
            # if itr < 300:
            #     img_input = img_l
            # else:
            #     img_input = img_h
            img_input = img_h

            img_jit = torch.roll(img, shifts=(off, off), dims=(2, 3))
            img_jit_l = torch.roll(img_l, shifts=(off, off), dims=(2, 3))
            img_jit_align = torch.roll(img_input, shifts=(off, off), dims=(2, 3))

            # Flipping
            flip = random.random() > 0.5
            if flip:
                img_jit = torch.flip(img_jit, dims=(3,))
                img_jit_l = torch.flip(img_jit_l, dims=(3,))
                img_jit_align = torch.flip(img_jit_align, dims=(3,))

            # Forward pass
            optimizer.zero_grad()
            p_model.zero_grad()

            output = p_model(img_jit)
            output_align = p_model(img_jit_align)

            T = 2.0  # Temperature coefficient
            teacher_output = (p_model(img_jit_l) / T).softmax(dim=-1).detach()
            student_output = (output_align / T).log_softmax(dim=-1)
            loss_kl = KL_Loss(student_output, teacher_output) * (T**2) 


            loss_align_semantic = criterion(output_align, pred)  # Make sure that the high-frequency enhanced plots still point to the correct class
            loss_align_consistency = F.mse_loss(output_align, output.detach())  # Ensure logical consistency with the original image
            # loss_hard = criterion(output_align, pred)
            loss_hard = 0.5 * loss_align_semantic + 0.5 * loss_align_consistency

            loss_align = args.align_weight * (torch.mean(loss_hard  + loss_kl )) / 1
            # print("loss_hard: ", loss_hard)


            if args.softlabel:
                loss_oh = 0
                # loss_soft = (torch.mean(soft_criterion(output, softlabel)) + criterion(output, pred)) / 2
                loss_soft = (torch.mean(soft_criterion(output, softlabel))) / 1
            else:
                loss_oh = criterion(output, pred)
                loss_soft = 0

            loss_tv = torch.norm(get_image_prior_losses(img_jit) - var_pred)


            coe_oh = args.coe_oh
            coe_sf = args.coe_sf
            coe_attn = args.coe_attn

            # attnmap loss
            loss_attnmaps = 0
            for itr_hook in range(len(attnhooks)):
                if 'swin' in args.model:
                    attention = attnhooks[itr_hook].feature
                    shapes = attnhooks[itr_hook].feature.shape
                    attnmap = attention[:, :, :, :]
                    # target = attnmap_pred[itr_hook].reshape(shapes[0], shapes[1], -1, 49)
                    # target = target.unsqueeze(2)
                    target = attnmap_pred[itr_hook].reshape(shapes[0], shapes[1], 1, 49)
                    target = target.expand(-1, -1, shapes[2], -1)

                    if itr_hook > len(attnhooks) // 2:
                        loss_attnmaps += ((itr_hook + 1) / len(attnhooks) * 1.0) * \
                                         F.mse_loss(attnmap, target, reduction='mean').float()
                else:
                    attention = attnhooks[itr_hook].feature  # [B,H,N,N]
                    attnmap = attention[:, :, 0:1, :]

                    # 确保 target 也是 [B, H, 1, 197]
                    target = attnmap_pred[:, itr_hook, :, :].reshape(attnmap.shape)

                    if itr_hook > len(attnhooks) // 2:
                        loss_attnmaps += ((itr_hook + 1) / 12 * 1.0) * \
                                         F.mse_loss(attnmap, target, reduction='mean').float()
                   

            loss_oh = coe_oh * loss_oh
            loss_soft = coe_sf * loss_soft
            loss_attnmaps = coe_attn * loss_attnmaps
            loss_tv = 0.05 * loss_tv
            total_loss = loss_oh + loss_tv + loss_soft + loss_attnmaps + loss_align


            if itr % 20 == 0:
                logging.info(
                    "[{}/2] {}/{}: total_loss={}, loss_oh={:.5f}, loss_tv={:.5f}, loss_soft={:.5f}, "
                    "loss_attnmaps={:.5f}".format(
                        lr_it + 1, itr, iterations_per_layer, total_loss, loss_oh, loss_tv, loss_soft,
                        loss_attnmaps))
                if itr < 300:
                  
                    delta = torch.zeros_like(img_jit).detach().requires_grad_(True)

                   
                    delta_optimizer = optim.Adam([delta], lr=1e-3)

                  
                    with torch.no_grad():
                        _ = p_model(img_jit)
                        feat_cle = p_model.features['norm'].detach()
                        output_cle = p_model(img_jit).detach()
                        output_cle_prob = F.softmax(output_cle, dim=1)

                    # Finding high frequency signals
                    for i in range(3):
                        p_model.zero_grad()
                        if delta.grad is not None:
                            delta.grad = None

                        output_adv = p_model(img_jit + delta)
                        feat_adv = p_model.features.get('norm', None)

                        if feat_adv is None:
                            print("Error: 'norm' layer not found in features!")
                            break

                        loss_ce = criterion(output_adv, pred)
                        loss_feat = F.mse_loss(feat_adv, feat_cle)

                        output_adv_logprob = F.log_softmax(output_adv, dim=1)
                        loss_kl = F.kl_div(output_adv_logprob, output_cle_prob, reduction='batchmean')

                        loss_tv_search = torch.norm(get_image_prior_losses(img_jit + delta) - var_pred)

                       
                        if i == 0:
                            current_adv_loss = loss_ce + loss_kl + 10 * loss_feat + 0.05 * loss_tv_search
                        else:
                            current_adv_loss = (3 * loss_ce + loss_kl + 10 * loss_feat + 0.05 * loss_tv_search) / 2


                        # Here only delta is updated, not img
                        current_adv_loss.backward()

                        with torch.no_grad():
                            grad_sign = delta.grad.sign()
                            delta.data = delta.data + 1e-3 * grad_sign
                            delta.data = torch.clamp(delta.data, -0.00784314, 0.00784314)
                            # print(delta.data)



                #     # High-frequency information is added to the image
                img_jit_adv = (img_jit + delta).detach().requires_grad_(True)

                output_adv = p_model(img_jit_adv)
                loss_ce_adv = criterion(output_adv, pred)

                # The total loss is recalculated, including the effect of high-frequency information
                if args.softlabel:
                    loss_oh_adv = 0
                    loss_soft_adv = torch.mean(soft_criterion(output_adv, softlabel)) / 1
                else:
                    loss_oh_adv = loss_ce_adv
                    loss_soft_adv = 0

                loss_tv_adv = torch.norm(get_image_prior_losses(img_jit_adv) - var_pred)

                loss_attnmaps_adv = 0
                for itr_hook in range(len(attnhooks)):
                    if 'swin' in args.model:
                        attention = attnhooks[itr_hook].feature
                        shapes = attnhooks[itr_hook].feature.shape
                        attnmap = attention[:, :, :, :]
                        # target = attnmap_pred[itr_hook].reshape(shapes[0], shapes[1], -1, 49)
                        target = attnmap_pred[itr_hook].reshape(shapes[0], shapes[1], 1, 49)
                        target = target.expand(-1, -1, shapes[2], -1) 
                        if itr_hook > len(attnhooks) // 2:
                            loss_attnmaps_adv += ((itr_hook + 1) / len(attnhooks) * 1.0) * \
                                                 F.mse_loss(attnmap, target, reduction='mean').float()
                    else:
                        attention = attnhooks[itr_hook].feature  # [B, H, 197, 197]
  
                        attnmap = attention[:, :, 0:1, :]

                        target = attnmap_pred[:, itr_hook, :, :].reshape(attnmap.shape)

                        if itr_hook > len(attnhooks) // 2:
                            loss_attnmaps += ((itr_hook + 1) / 12 * 1.0) * \
                                             F.mse_loss(attnmap, target, reduction='mean').float()
                        # attention = attnhooks[itr_hook].feature
                        # attnmap = attention[:, :, 0, :]
                        # if itr_hook > len(attnhooks) // 2:
                        #     loss_attnmaps_adv += ((itr_hook + 1) / 12 * 1.0) * \
                        #                          F.mse_loss(attnmap, attnmap_pred[:, itr_hook, :, :],
                        #                                     reduction='mean').float()

                loss_oh_adv = coe_oh * loss_oh_adv
                loss_soft_adv = coe_sf * loss_soft_adv
                loss_attnmaps_adv = coe_attn * loss_attnmaps_adv
                loss_tv_adv = 0.05 * loss_tv_adv
                total_loss_adv = loss_oh_adv + loss_tv_adv + loss_soft_adv + loss_attnmaps_adv
                total_loss_adv.backward()

            else:
                # Normal training phase
                total_loss.backward()

            # Do image update

            # total_loss.backward()
            loss_history['total_loss'].append(get_val(total_loss))
            loss_history['loss_oh'].append(get_val(loss_oh))
            loss_history['loss_soft'].append(get_val(loss_soft))
            loss_history['loss_tv'].append(get_val(loss_tv))
            loss_history['loss_attnmaps'].append(get_val(loss_attnmaps))
            loss_history['loss_align'].append(get_val(loss_align))
            loss_history['loss_kl'].append(get_val(loss_kl))
            loss_history['loss_hard'].append(get_val(loss_hard))
            optimizer.step()
            del output, output_align, total_loss
            if 'total_loss_adv' in locals():
                del total_loss_adv

            if args.softlabel:
                img_cropped = []
                img_cropped_label = []
                softlabel_cropped = []
                for i in range(len(pred)):
                    if len(softboxes[i]) == 1:
                        continue
                    for j in range(len(softboxes[i])):
                        current = img[i]
                        box = softboxes[i][j]
                        cropped_label = softtagets[i][j]
                        cropped_softlabel = softtagets_softlabel[i][j]

                        top = box[0]
                        left = box[1]
                        box_height = box[2]
                        box_width = box[3]
                        crop_width = np.random.randint(max(box_width - 30, 10), box_width - 1)
                        crop_height = np.random.randint(max(box_height - 30, 10), box_height - 1)
                        crop_top = np.random.randint(top, top + box_height - crop_height)
                        crop_left = np.random.randint(left, left + box_width - crop_width)

                        img_cropped.append(
                            resize_transform(transforms.functional.crop(current, top=crop_top, left=crop_left,
                                                                        height=crop_height, width=crop_width)))
                        img_cropped_label.append(cropped_label)
                        softlabel_cropped.append(cropped_softlabel)

                if len(img_cropped) > 0:
                    img_cropped_label = torch.stack(img_cropped_label).cuda()
                    softlabel_cropped = torch.stack(softlabel_cropped).cuda()
                    img_cropped = torch.stack(img_cropped).cuda()

                    # Forward pass
                    optimizer.zero_grad()
                    p_model.zero_grad()

                    loss_attnmaps_cropped = 0
                    img_jit_cropped = torch.roll(img_cropped, shifts=(off, off), dims=(2, 3))
                    if flip:
                        img_jit_cropped = torch.flip(img_jit_cropped, dims=(3,))
                    output_cropped = p_model(img_jit_cropped)

                    if args.softlabel:
                        loss_oh_cropped = 0
                        loss_soft_cropped = (torch.mean(soft_criterion(output_cropped, softlabel_cropped))) / 1
                    else:
                        loss_oh_cropped = criterion(output_cropped, img_cropped_label)
                        loss_soft_cropped = 0

                    loss_tv_cropped = torch.norm(get_image_prior_losses(img_jit_cropped) - var_pred_cropped)

                  
                    for itr_hook in range(len(attnhooks)):
                        if 'swin' in args.model:
                            attention = attnhooks[itr_hook].feature
                            shapes = attention.shape  # [B_cropped, H, 49, 49]
                            attnmap = attention

                            target = attnmap_pred_cropped[itr_hook].reshape(shapes[0], shapes[1], 1, 49)
                            target = target.expand(-1, -1, shapes[2], -1)

                            if itr_hook > len(attnhooks) // 2:
          
                                loss_attnmaps_cropped += ((itr_hook + 1) / len(attnhooks) * 1.0) * \
                                                         F.mse_loss(attnmap, target, reduction='mean').float()
                        else:
                            attention = attnhooks[itr_hook].feature  # [B,H,N,N]
                            attnmap = attention[:, :, 0, :]  # [B,H,N,N]->[B,H,N]
                            if itr_hook > len(attnhooks) // 2:
                                loss_attnmaps_cropped += ((itr_hook + 1) / 12 * 1.0) * \
                                                         F.mse_loss(attnmap, attnmap_pred_cropped[:, itr_hook, :, :],
                                                                    reduction='mean').float()

                    loss_oh_cropped = coe_oh * loss_oh_cropped
                    loss_soft_cropped = coe_sf * loss_soft_cropped
                    loss_tv_cropped = 0.05 * loss_tv_cropped
                    loss_attnmaps_cropped = coe_attn * loss_attnmaps_cropped
                    total_loss = loss_oh_cropped + loss_soft_cropped + loss_tv_cropped + \
                                 loss_attnmaps_cropped

                    if itr % 20 == 0:
                        logging.info(
                            "[{}/2] {}/{}: total_loss_cropped={:.5f}, loss_oh_cropped={:.5f}, loss_soft_cropped={:.5f}"
                            "loss_tv_cropped={:5f},"
                            "loss_attnmaps_cropped={:5f},".format(
                                lr_it + 1, itr, iterations_per_layer, total_loss,
                                loss_oh_cropped, loss_soft_cropped,
                                loss_tv_cropped, loss_attnmaps_cropped))
                    
                    # Do image update
                    total_loss.backward()
                    optimizer.step()
                
                else:
                    pass

            # Clip color outliers
            img.data = clip(img.data)
    if args.softlabel:
        return img.detach(), softboxes, softtagets,loss_history
    else:
        return img.detach(), None, None,loss_history


class AttentionMap:
    def __init__(self, module, batchsize=0, cropped_number=0):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.batchsize = batchsize
        self.cropped_number = cropped_number
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()



class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """

    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss




def get_image_prior_losses(inputs_jit):
    # Compute total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2


def clip(image_tensor, use_fp16=False):
    # Adjust the input based on mean and variance
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
        # image_tensor[:, c] = torch.clamp(image_tensor[:, c], 0, 1)
    return image_tensor


def img_reverse(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[c] = image_tensor[c] * s + m
    image_tensor = image_tensor * 255
    return image_tensor
    # image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 255)


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)
