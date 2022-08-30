
import sys, os
sys.path.append("/root/rjchen/workspace/SR")
import os, math, random
from collections import OrderedDict
from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from supernet.swinir3 import SwinIR
from dataset import SRDataset, RandomHorizontalFlip, RandomRotation, ToTensor

DISABLE = False

def param_grad(model):
    grads = OrderedDict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads

def cosine_similarity(grad1, grad2):
    sims = OrderedDict()
    names = grad1.keys()
    for name in names:
        g1 = grad1[name]
        g2 = grad2[name]
        cossim = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=-1, eps=1e-08)
        sims[name] = cossim.item()
    
    return sims

def backward(epoch, iters, model, dataloader, criterion, optimizer):
    dataiter = iter(dataloader)
    model.train()
    grad0 = None
    with tqdm(range(iters), desc=f"Train({epoch:>2d})", ncols=100, disable=DISABLE) as t:
        for it in t:
            try:
                lr, hr = next(dataiter)
            except:
                dataiter = iter(dataloader)
                lr, hr = next(dataiter)

            lr, hr = lr.to(device), hr.to(device)
            
            # if it%4 == 0:
            #     model.sample("max")
            #     output = model(lr)
            #     loss = criterion(output, hr)
            # elif it%4 == 3:
            #     model.sample("min")
            #     output = model(lr)
            #     loss = criterion(output, hr)
            # else:
            #     model.sample("random")
            #     output = model(lr)
            #     loss = criterion(output, hr)
            # loss.backward()
            # if it % 4 == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            model.sample("random")
            output = model(lr)
            loss = criterion(output, hr)

            optimizer.zero_grad()
            loss.backward()
            grad = param_grad(model)
            optimizer.step()

            if grad0 is not None:
                sim = cosine_similarity(grad0, grad)
                print(sim)
            grad0 = grad


            


def main(args):
    init_kwargs = {
        'search_embed_dim': [16, 32, 48, 64],
        'search_num_feat': [16, 32, 48, 64],
        'search_layers': [0, 1, 2, 3, 4],
        'search_num_heads': [2, 3, 4],
        'search_mlp_ratio': [1.0, 1.5, 2.0],
        'upscale': args.scale,
        'img_size': (64, 64),
        'patch_size': 1,
        'window_size': 4,
    }
    config = {
        "model": "SwinIRv3", 
        "desc": "random sample"
    }
    config.update(init_kwargs)
    config.update(vars(args))
    model = SwinIR(**init_kwargs)

    # if isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
    #     model.load_state_dict(torch.load(args.pretrained))

    model.to(device)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    transform = [
        RandomRotation([0, 90, 180, 270]), 
        RandomHorizontalFlip(0.5), 
        ToTensor()
    ]
    train_set = SRDataset(args.train_data, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    backward(0, 8, model, train_loader, criterion, optimizer)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=False, default="/root/rjchen/data/SR/DIV2K_train_2.h5", help='train data folder')
    parser.add_argument('--valid_data', type=str, required=False, default="/root/rjchen/data/SR/DIV2K_valid_2.h5", help='valid data folder')
    parser.add_argument('--pretrained', type=str, required=False, default=None, help='pretrained weights')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--upsample', action="store_true")
    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=6 python grid.py --pretrained /root/rjchen/workspace/outputs/SwinIR3dev/full_sandwich_X2/best.pth
CUDA_VISIBLE_DEVICES=4 python train.py --suffix depth0_dim32_min
CUDA_VISIBLE_DEVICES=7 python train.py --suffix sandwich --out ../outputs/SwinIR3dev

CUDA_VISIBLE_DEVICES=7 python train.py --layers 4 --suffix depth4_dim32_pretrained --out ../outputs/SwinIR2 --pretrained /root/rjchen/workspace/outputs/SwinIR2/depth4_dim32_fixed_X2/best.pth
CUDA_VISIBLE_DEVICES=6 python train.py --layers 4 --suffix depth4_dim32 --out ../outputs/SwinIR2
CUDA_VISIBLE_DEVICES=7 python train.py --suffix depth2_dim32 --out ../outputs/SwinIR2
CUDA_VISIBLE_DEVICES=6 python train.py --suffix depth0_dim32 --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs/SwinIR2

CUDA_VISIBLE_DEVICES=7 python train.py --suffix dev --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix fsrcnn_l1_supernet --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=2 python train.py --suffix fsrcnn_l1_supernet_upsample --upsample --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
CUDA_VISIBLE_DEVICES=7 python train.py --suffix super_l1_d56 --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs
"""