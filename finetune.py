from cProfile import label
import os, math, random

from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from model.fsrcnn import FSRCNN
from supernet.fsrcnn import FSRCNN
from dataset import SRDataset, RandomHorizontalFlip, RandomRotation, ToTensor
from utils import AverageMeter, batch_psnr

DISABLE = False

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

def train(epoch, iters, model, dataloader, criterion, optimizer, scheduler):
    Loss = AverageMeter()
    PSNR = AverageMeter()

    dataiter = iter(dataloader)
    model.train()
    with tqdm(range(iters), desc=f"Train({epoch:>2d})", ncols=100, disable=DISABLE) as t:
        for it in t:
            try:
                lr, hr = next(dataiter)
            except:
                dataiter = iter(dataloader)
                lr, hr = next(dataiter)
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)

            output = model(lr)
            loss = criterion(output, hr)
            psnr = batch_psnr(output.detach().clamp(0.0, 1.0), hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            Loss.update(loss.item(), batch_size)
            PSNR.update(psnr.item(), batch_size)

            lr = scheduler.get_last_lr()[0]

            t.set_postfix({"Loss": f"{Loss.item():.4f}", "PSNR": f"{PSNR.item():.2f} dB", "LR": f"{lr:.5f}"})

    return Loss.item(), PSNR.item()

from collections import OrderedDict
import torch.nn as nn


def main(args):
    d, s, m = 56, 12, 4
    model = FSRCNN(args.scale1, 3, d, 12, 4, args.upsample)
    model.load_state_dict(torch.load(args.weight))
    model.deconv = nn.Sequential(OrderedDict([
        ("conv1", nn.Conv2d(d, d//2*args.scale2**2, kernel_size=3, stride=1, padding=1)),
        ("pixelshuffle", nn.PixelShuffle(args.scale2)),
        ("conv2", nn.Conv2d(d//2, d//2, kernel_size=3, stride=1, padding=1)),
        ("prelu", nn.PReLU(d//2)),
        ("conv3", nn.Conv2d(d//2, 3, kernel_size=3, stride=1, padding=1)),
    ]))
    for name, param in model.named_parameters():
        if "deconv" not in name:
            param.requires_grad = False
    model.to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    warm_up = 0
    lr_min = 0.01
    T_max = args.total_step
    lr_lambda = lambda i: i / warm_up if i<warm_up else lr_min + (1-lr_min)*(1.0+math.cos((i-warm_up)/(T_max-warm_up)*math.pi))/2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    transform = [
        RandomRotation([0, 90, 180, 270]), 
        RandomHorizontalFlip(0.5), 
        ToTensor()
    ]
    train_set = SRDataset(args.train_data, transform=transform)
    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    best_psnr = 0
    psnr = valid(model, valid_loader)
    for epoch in range(args.total_step//args.valid_step):
        loss, _ = train(epoch, args.valid_step, model, train_loader, criterion, optimizer, scheduler)
        psnr = valid(model, valid_loader)

        if psnr > best_psnr:
            best_psnr = psnr
        #     torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
        # torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
    
    print(psnr, best_psnr)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--suffix', type=str, required=True, help='exp suffix')
    parser.add_argument('--weight', type=str, required=True, help='')
    parser.add_argument('--train_data', type=str, required=True, help='train data folder')
    parser.add_argument('--valid_data', type=str, required=True, help='valid data folder')
    parser.add_argument('--out', type=str, required=True, help='output folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale1', type=int, default=2)
    parser.add_argument('--scale2', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--total_step', type=int, default=2**15)
    parser.add_argument('--valid_step', type=int, default=2**10)
    parser.add_argument('--upsample', action="store_true")
    args = parser.parse_args()

    # args.out = os.path.join(args.out, f"{args.suffix}_X{args.scale}")
    # os.makedirs(args.out, exist_ok=True)

    main(args)

"""
CUDA_VISIBLE_DEVICES=2 python finetune.py --train_data /root/rjchen/data/SR/DIV2K_train_3.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_3.h5 --out ../outputs --valid_step 1024 --total_step 16384 --scale1 2 --scale2 3 --weight /root/rjchen/workspace/outputs/dev_X2/best.pth

CUDA_VISIBLE_DEVICES=7 python train.py --suffix dev --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix fsrcnn_l1_supernet --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=2 python train.py --suffix fsrcnn_l1_supernet_upsample --upsample --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix super_l1_d56 --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
"""