import os, math, random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model.fsrcnn import FSRCNN
# from model.cran import CARN
# from supernet.fsrcnn import FSRCNN
from dataset import SRDataset, RandomHorizontalFlip, RandomRotation, ToTensor
from utils import AverageMeter, batch_psnr

DISABLE = False


def valid(model, dataloader, criterion):
    PSNR = AverageMeter()
    PSNR0 = AverageMeter()
    Loss = AverageMeter()
    Loss0 = AverageMeter()
    with tqdm(dataloader, total=len(dataloader), desc="Valid", ncols=120, disable=DISABLE) as t:
        for lr, hr in t:
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            lr.requires_grad = True
            optimizer = optim.Adam([lr], lr=0.0001)
            for _ in range(32):
                out = model(lr)
                loss = criterion(out, hr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # grad = torch.autograd.grad(loss, [lr])[0].detach()
                # lr = lr.detach() - (0.002/(_+1))*grad.sign()
                # lr = lr.clamp(0.0, 1.0)
                psnr = batch_psnr(out.detach().clamp(0.0, 1.0), hr)
                if _ == 0:
                    psnr = batch_psnr(out.detach().clamp(0.0, 1.0), hr)
                    PSNR0.update(psnr.item(), batch_size)
                    Loss0.update(loss.item(), batch_size)
                # print(_, loss.item(), psnr.item())
            
            psnr = batch_psnr(out, hr, True)
            PSNR.update(psnr.item(), batch_size)
            Loss.update(loss.item(), batch_size)
            t.set_postfix({"PSNR": f"{PSNR.item():.2f} dB", "PSNR0": f"{PSNR0.item():.2f}", "Loss": f"{Loss.item():.4f} dB", "Loss0": f"{Loss0.item():.4f}"})
            # break
    return PSNR.item()


def main(args):
    d, s, m = args.d, 12, 4
    model = FSRCNN(args.scale, 3, d, 12, 4, args.upsample).to(device)
    model.load_state_dict(torch.load(f"../outputs/fsrcnn_l1_d56{'_upsample' if args.upsample else ''}_X2/best.pth"))
    for p in model.parameters():
        p.requires_grad = False
    criterion = nn.L1Loss(reduction='mean').to(device)
    transform = [
        RandomRotation([0, 90, 180, 270]), 
        RandomHorizontalFlip(0.5), 
        ToTensor()
    ]
    train_set = SRDataset(args.train_data, transform=transform)
    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best_psnr = 0
    psnr = valid(model, valid_loader, criterion)


    print(psnr, best_psnr)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='train data folder')
    parser.add_argument('--valid_data', type=str, required=True, help='valid data folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--total_step', type=int, default=2**16)
    parser.add_argument('--valid_step', type=int, default=2**10)
    parser.add_argument('--upsample', action="store_true")
    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=7 python fgsm.py --upsample --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5
CUDA_VISIBLE_DEVICES=2 python train.py --suffix fsrcnn_l1_supernet_upsample --upsample --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix super_l1_d56 --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
"""