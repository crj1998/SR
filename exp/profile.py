
import os, math, random

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from model import FSRCNN
from supernet.fsrcnn import FSRCNN
from supernet.fsrcnn import Conv2d, PReLU, ConvTranspose2d
from dataset import SRDataset
from utils import AverageMeter, batch_psnr

DISABLE = True

@torch.no_grad()
def valid(model, dataloader):
    PSNR = AverageMeter()
    with tqdm(dataloader, total=len(dataloader), desc="Valid", ncols=100, disable=DISABLE) as t:
        for lr, hr in t:
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            psnr = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr)
            PSNR.update(psnr.item(), batch_size)
            t.set_postfix({"PSNR": f"{PSNR.item():.2f} dB"})
    return PSNR.item()


from thop import profile
from thop.vision.basic_hooks import count_convNd, count_prelu
import json

def main(args):
    d, s, m = args.d, 12, 4
    model = FSRCNN(args.scale, 3, d, s, m)
    model.load_state_dict(torch.load(args.weight))

    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    custom_ops = {
        Conv2d: count_convNd,
        ConvTranspose2d: count_convNd,
        PReLU: count_prelu,
    }

    info = {"name": "FSRCNN", "profile": []}

    x = torch.randn(1, 3, 64, 64)
    print(f"{'d':>2s} | {'Params':>7s} | {'MACs':>7s} | {'PSNR':>5s}")
    for d in range(3, 57, 1):
        if hasattr(model, "set_sample_config"):
            model.set_sample_config(d)
        model = model.to(device)
        psnr = valid(model, valid_loader)

        model = model.cpu()
        macs, _ = profile(model, inputs=(x, ), custom_ops=custom_ops, verbose=False)
        params = model.get_params()

        print(f"{d:>2d} | {params/10**3:7.3f} | {macs/10**6:7.3f} | {psnr:5.2f}")
        info["profile"].append({"params": int(params), "macs": int(macs), "psnr": psnr})

    with open("../outputs/cache/anchor_profile.json", "w") as f:
        json.dump(info, f)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help='mdoel weight')
    parser.add_argument('--valid_data', type=str, required=True, help='valid data folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    main(args)

"""
python exp.py --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --weight /root/rjchen/workspace/outputs/base_l1_anchor_d112_X2/best.pth
"""