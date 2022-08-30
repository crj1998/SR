from cProfile import label
import os, math, random

from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

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
    # i = 0
    with tqdm(dataloader, total=len(dataloader), desc="Valid", ncols=100, disable=DISABLE) as t:
        for lr, hr in t:
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            psnr = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr)
            PSNR.update(psnr.item(), batch_size)
            t.set_postfix({"PSNR": f"{PSNR.item():.2f} dB"})
            # i += 1
            # if i > 32:
            #     break
    return PSNR.item()

def train(epoch, iters, model, dataloaders, criterion, optimizer, scheduler):
    Loss = AverageMeter()
    PSNR = AverageMeter()

    dataiters = [iter(dataloader) for dataloader in dataloaders]
    model.train()
    with tqdm(range(iters), desc=f"Train({epoch:>2d})", ncols=100, disable=DISABLE) as t:
        for it in t:
            idx = random.choice([0, 1, 1, 2, 2, 2])
            try:
                lr, hr = next(dataiters[idx])
            except:
                dataiters[idx] = iter(dataloaders[idx])
                lr, hr = next(dataiters[idx])
            if hasattr(model, "set_sample_config"):
                model.set_sample_config([2, 3, 4][idx])

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


def main(args):
    d, s, m = args.d, 12, 4
    model = FSRCNN(4, 3, 56, 12, 4, args.upsample)

    model.to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
    train_set_2 = SRDataset("/root/rjchen/data/SR/DIV2K_train_2.h5", transform=transform)
    train_set_3 = SRDataset("/root/rjchen/data/SR/DIV2K_train_3.h5", transform=transform)
    train_set_4 = SRDataset("/root/rjchen/data/SR/DIV2K_train_4.h5", transform=transform)
    valid_set_2 = SRDataset("/root/rjchen/data/SR/DIV2K_valid_2.h5", transform=T.ToTensor())
    valid_set_3 = SRDataset("/root/rjchen/data/SR/DIV2K_valid_3.h5", transform=T.ToTensor())
    valid_set_4 = SRDataset("/root/rjchen/data/SR/DIV2K_valid_4.h5", transform=T.ToTensor())
    train_loader_2 = DataLoader(train_set_2, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_loader_3 = DataLoader(train_set_3, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_loader_4 = DataLoader(train_set_4, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader_2 = DataLoader(valid_set_2, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader_3 = DataLoader(valid_set_3, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader_4 = DataLoader(valid_set_4, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_loaders = [train_loader_2, train_loader_3, train_loader_4]
    valid_loaders = [valid_loader_2, valid_loader_3, valid_loader_4]
    best_psnr = 0

    for epoch in range(args.total_step//args.valid_step):
        loss, _ = train(epoch, args.valid_step, model, train_loaders, criterion, optimizer, scheduler)
        # for idx, s in enumerate([2, 3, 4]):
        #     model.set_sample_config(s)
        #     psnr = valid(model, valid_loaders[idx])

        # if psnr > best_psnr:
        #     best_psnr = psnr
        #     torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
        # torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
    
    # print(psnr, best_psnr)

    model.set_sample_config(2)
    psnr = valid(model, valid_loader_2)
    model.set_sample_config(3)
    psnr = valid(model, valid_loader_3)
    model.set_sample_config(4)
    psnr = valid(model, valid_loader_4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, required=True, help='exp suffix')
    parser.add_argument('--train_data', type=str, help='train data folder')
    parser.add_argument('--valid_data', type=str, help='valid data folder')
    parser.add_argument('--out', type=str, required=True, help='output folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--total_step', type=int, default=2**15)
    parser.add_argument('--valid_step', type=int, default=2**10)
    parser.add_argument('--upsample', action="store_true")
    args = parser.parse_args()

    args.out = os.path.join(args.out, f"{args.suffix}_X{args.scale}")
    os.makedirs(args.out, exist_ok=True)

    main(args)

"""
CUDA_VISIBLE_DEVICES=0 python multiscale.py --suffix fsrcnn_multi --out ../outputs --valid_step 1024 --total_step 32768

CUDA_VISIBLE_DEVICES=7 python train.py --suffix dev --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix fsrcnn_l1_supernet --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=2 python train.py --suffix fsrcnn_l1_supernet_upsample --upsample --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix super_l1_d56 --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
"""