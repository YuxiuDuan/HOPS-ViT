import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import time
import os
import random
import torch
import pickle
import logging
import sys
import numpy as np
from generate_data import generate_data_mulitTarget


def setup_logging(log_file):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def Time2Str():
    sec = time.time()
    tm = time.localtime(sec)
    time_str = '{}{:02d}{:02d}_{:02d}_{:02d}'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min)
    return time_str


def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_args_parser():
    parser = argparse.ArgumentParser(description="PSAQ-ViT", add_help=False)
    parser.add_argument("--model", default="vit_base",
                        choices=['vit_small', 'vit_base','vit_tiny', 'deit_tiny',
                                 'deit_small', 'deit_base', 'swin_tiny', 'swin_base',
                                 'swin_small'],
                        help="model")
    parser.add_argument("--calib_batchsize", default=32,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument('--softlabel', action='store_true', help='enable soft label')
    parser.add_argument('--save_fake', action='store_true', help='save fake data')
    parser.add_argument("--name", default=None, type=str, help="experiment name")

    parser.add_argument("--coe_oh", default=0, type=float, help="seed")
    parser.add_argument("--coe_sf", default=1, type=float, help="seed")
    parser.add_argument("--coe_attn", default=100, type=float, help="seed")
    parser.add_argument("--align_weight", default=1, type=float, help="seed")

    return parser


def setup_experiment(args):
    """
    Set up the experiment directory, logging, and seed.
    """
    seed(args.seed)
    args.results_dir = f'./results/{args.model}/generate/{Time2Str()}'
    os.makedirs(args.results_dir, exist_ok=True)
    setup_logging(os.path.join(args.results_dir, 'log.txt'))
    logging.info("Experiment name: %s", args.name)
    logging.info("Running arguments: %s", args)



def generate_and_save_fake_data(args, device):
    """
    Generate and save fake data based on the provided arguments.
    """
    pred = torch.LongTensor([random.randint(0, 99) for _ in range(args.calib_batchsize)]).to(device)
    logging.info("Generating and saving fake data...")

    imgs_list, softboxes_list, softtargets_list = [], [], []
    all_loss_histories = []

    batch_divisor = 4 if 'swin_base' in args.model else 16
    # if 'base' in args.model:
    #     batch_divisor = 4  # 原来是 16 或 8，改为 4
    # else:
    #     batch_divisor = 16
    for i in range(args.calib_batchsize // batch_divisor):
        print("第{}轮".format(i))
        batch_indices = slice(batch_divisor * i, batch_divisor * (i + 1))
        logging.info("Hard label: %s", pred[batch_indices])
        imgs, softboxes, softtargets, loss_hist = generate_data_mulitTarget(args, pred=pred[batch_indices])
        imgs_list.extend(imgs.cpu())
        all_loss_histories.append(loss_hist)
        if args.softlabel:
            softboxes_list.extend(softboxes)
            softtargets_list.extend(softtargets)

    save_generated_data(args, imgs_list, pred, softboxes_list, softtargets_list, all_loss_histories)
# def generate_and_save_fake_data(args, device):
#     """
#     Generate and save fake data based on the provided arguments.
#     """
#     logging.info("Generating and saving fake data...")
#     pred = torch.LongTensor([random.randint(0, 99) for _ in range(args.calib_batchsize)]).to(device)
#     imgs_list, softboxes_list, softtargets_list = [], [], []
#
#
#     batch_divisor = 4 if 'swin_base' in args.model else 4
#     # if 'base' in args.model:
#     #     batch_divisor = 4  # 原来是 16 或 8，改为 4
#     # else:
#     #     batch_divisor = 16
#     for i in range(args.calib_batchsize // batch_divisor):
#         print("第{}轮".format(i))
#         batch_indices = slice(batch_divisor * i, batch_divisor * (i + 1))
#         logging.info("Hard label: %s", pred[batch_indices])
#         imgs, softboxes, softtargets = generate_data_mulitTarget(args, pred=pred[batch_indices])
#         imgs_list.extend(imgs.cpu())
#
#         if args.softlabel:
#             softboxes_list.extend(softboxes)
#             softtargets_list.extend(softtargets)
#
#     save_generated_data(args, imgs_list, pred, softboxes_list, softtargets_list)


def save_generated_data(args, imgs_list, pred, softboxes_list, softtargets_list,all_loss_histories):
    """
    Save generated fake data to the disk.
    """
    savepath = f'./fake_data/{args.model}'
    os.makedirs(savepath, exist_ok=True)

    imgs = torch.stack(imgs_list).cpu().numpy()
    pred = pred.cpu().numpy()
    logging.info("Image shape is %s", imgs.shape)
    save_pickle(os.path.join(savepath, f'align--{args.align_weight}-'
                                       f'coe_oh-{args.coe_oh}-'
                                       f'coe_sf-{args.coe_sf}-'
                                       f'coe_attn-{args.coe_attn}-'
                                       f'-bs{args.calib_batchsize}_data.pickle'), imgs)
    save_pickle(os.path.join(savepath, f'align--{args.align_weight}-'
                                       f'coe_oh-{args.coe_oh}-'
                                       f'coe_sf-{args.coe_sf}-'
                                       f'coe_attn-{args.coe_attn}-'
                                       f'-bs{args.calib_batchsize}_label.pickle'), pred)
    loss_filename = f'align--{args.align_weight}-coe_oh-{args.coe_oh}-coe_sf-{args.coe_sf}-loss_history.pickle'
    save_pickle(os.path.join(savepath, loss_filename), all_loss_histories)
    logging.info(f"Loss history saved to {loss_filename}")

    if args.softlabel:
        save_pickle(os.path.join(savepath, f'align--{args.align_weight}-'
                                           f'coe_oh-{args.coe_oh}-'
                                           f'coe_sf-{args.coe_sf}-'
                                           f'coe_attn-{args.coe_attn}-'
                                           f'-bs{args.calib_batchsize}_boxes.pickle'), softboxes_list)
        save_pickle(os.path.join(savepath, f'align--{args.align_weight}-'
                                           f'coe_oh-{args.coe_oh}-'
                                           f'coe_sf-{args.coe_sf}-'
                                           f'coe_attn-{args.coe_attn}-'
                                           f'-bs{args.calib_batchsize}_softtargets.pickle'), softtargets_list)
# def save_generated_data(args, imgs_list, pred, softboxes_list, softtargets_list):
#     """
#     Save generated fake data to the disk.
#     """
#     savepath = f'./fake_data/{args.model}'
#     os.makedirs(savepath, exist_ok=True)
#
#     imgs = torch.stack(imgs_list).cpu().numpy()
#     pred = pred.cpu().numpy()
#     logging.info("Image shape is %s", imgs.shape)
#     save_pickle(os.path.join(savepath, f'coe_oh-{args.coe_oh}-'
#                                        f'coe_sf-{args.coe_sf}-'
#                                        f'coe_attn-{args.coe_attn}-'
#                                        f'-bs{args.calib_batchsize}_data.pickle'), imgs)
#     save_pickle(os.path.join(savepath, f'coe_oh-{args.coe_oh}-'
#                                        f'coe_sf-{args.coe_sf}-'
#                                        f'coe_attn-{args.coe_attn}-'
#                                        f'-bs{args.calib_batchsize}_label.pickle'), pred)
#
#
#     if args.softlabel:
#         save_pickle(os.path.join(savepath, f'coe_oh-{args.coe_oh}-'
#                                            f'coe_sf-{args.coe_sf}-'
#                                            f'coe_attn-{args.coe_attn}-'
#                                            f'-bs{args.calib_batchsize}_boxes.pickle'), softboxes_list)
#         save_pickle(os.path.join(savepath, f'coe_oh-{args.coe_oh}-'
#                                            f'coe_sf-{args.coe_sf}-'
#                                            f'coe_attn-{args.coe_attn}-'
#                                            f'-bs{args.calib_batchsize}_softtargets.pickle'), softtargets_list)


def save_pickle(filepath, data):
    """
    Save data to a pickle file.
    """
    with open(filepath, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    setup_experiment(args)
    device = torch.device(args.device)

    if args.save_fake:
        generate_and_save_fake_data(args, device)
        return


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser('SFP', parents=[get_args_parser()])
    args = parser.parse_args()
    weight_list = [ 1.2, 1.4, 1.6, 1.8, 2.0]
    for weight in weight_list:

        args.align_weight = weight
        print("args.align_weight", args.align_weight)
        main(args)
        total_train_time = time.time() - start_time
        logging.info("Training time: %dh:%dm:%ds", total_train_time // 3600, (total_train_time % 3600) // 60, total_train_time % 60)
