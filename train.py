from cProfile import label
import os, math, random

from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from model.fsrcnn import FSRCNN
# from model.cran import CARN
# from model.swinir import SwinIR
# from model.convformer import Convformer
# from supernet.swinir import SwinIR
# from supernet.swinir2 import SwinIR
from supernet.swinir3 import SwinIR
# from supernet.fsrcnn import FSRCNN
from dataset import SRDataset, RandomHorizontalFlip, RandomRotation, ToTensor
from utils import AverageMeter, batch_psnr, CSVwriter

DISABLE = False

@torch.no_grad()
def valid(model, dataloader, breakpoint=None):
    PSNR = AverageMeter()
    with tqdm(enumerate(dataloader), total=breakpoint or len(dataloader), desc="Valid", ncols=80, disable=DISABLE) as t:
        for i, (lr, hr) in t:
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            psnr = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr)
            PSNR.update(psnr.item(), batch_size)
            t.set_postfix({"PSNR": f"{PSNR.item():.2f} dB"})
            if breakpoint is not None and i >= breakpoint:
                break

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

            learning_rate = scheduler.get_last_lr()[0]
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            
            if it%4 == 0:
                model.sample("max")
                output = model(lr)
                loss = criterion(output, hr)
            elif it%4 == 3:
                model.sample("min")
                output = model(lr)
                loss = criterion(output, hr)
            else:
                model.sample("random")
                output = model(lr)
                loss = criterion(output, hr)
            loss.backward()
            if it % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
                for _ in range(4):
                    scheduler.step()

            # model.sample("random")
            # output = model(lr)
            # loss = criterion(output, hr)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            
            psnr = batch_psnr(output.detach().clamp(0.0, 1.0), hr)
            Loss.update(loss.item(), batch_size)
            PSNR.update(psnr.item(), batch_size)
            

            t.set_postfix({"Loss": f"{Loss.item():.4f}", "PSNR": f"{PSNR.item():.2f} dB", "LR": f"{learning_rate:.5f}"})

    return Loss.item(), PSNR.item()


def main(args):
    import wandb
    # init_kwargs = {
    #     'search_embed_dim': [16, 24, 32],
    #     'search_num_feat': [16, 24, 32],
    #     'search_layers': [0, 1, 2, 3, 4],
    #     'search_num_heads': [2, 3, 4],
    #     'search_mlp_ratio': [1.0, 1.5, 2.0],
    #     'upscale': args.scale,
    #     'img_size': (64, 64),
    #     'patch_size': 1,
    #     'window_size': 4,
    # }
    # init_kwargs = {
    #     'search_embed_dim': [16, 32, 48, 64],
    #     'search_num_feat': [16, 32, 48, 64],
    #     'search_layers': [0, 1, 2, 3, 4],
    #     'search_num_heads': [4, 6, 8],
    #     'search_mlp_ratio': [1.0, 1.5, 2.0],
    #     'upscale': args.scale,
    #     'img_size': (64, 64),
    #     'patch_size': 1,
    #     'window_size': 4,
    # }
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
    wandb.init(project="SR", entity="maze", config=config, name=args.suffix)

    # model = FSRCNN(args.scale, 3, d, 12, 4, args.upsample)
    # model = SwinIR(
    #     upscale=args.scale, img_size=(64, 64), patch_size=1, embed_dim=32, num_feat=32,
    #     window_size=4, depths=[2, 2, 2, 2], num_heads=[4, 4, 4, 4], mlp_ratio=2.0
    # )
    # model = SwinIR(
    #     upscale=args.scale, img_size=(64, 64), patch_size=1, embed_dim=64, num_feat=64,
    #     window_size=4, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], mlp_ratio=2.0
    # )
    # swinir2
    # model = SwinIR(
    #     upscale=args.scale, embed_dim=args.embed_dim, num_feat=args.num_feat, 
    #     img_size=(64, 64), patch_size=1, window_size=4, 
    #     depths=[2]*4, num_heads=[4]*4, mlp_ratio=2.0
    # )
    # swinir3
    # model = SwinIR(
    #     search_embed_dim=search_embed_dim, search_num_feat=search_num_feat, 
    #     search_layers=search_layers, search_num_heads=search_num_heads, search_mlp_ratio=search_mlp_ratio, 
    #     upscale=args.scale, img_size=(64, 64), patch_size=1, window_size=4
    # )
    model = SwinIR(**init_kwargs)

    # [4]*4, 38.23
    if isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
        model.load_state_dict(torch.load(args.pretrained))

    # model = Convformer(args.scale, img_size=64, window_size=4, num_feat=64, embed_dim=32, depths=3, num_heads=4, drop_path_rate=0.1)
    model.to(device)
    criterion = nn.L1Loss(reduction='mean')
    # criterion = nn.MSELoss(reduction='mean').to(device)
    # optimizer = optim.Adam([
    #     {"params": [p for n, p in model.named_parameters() if "deconv" not in n]},
    #     {"params": model.deconv.parameters(), "lr": args.lr * 0.1}
    # ], lr=args.lr)
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
    train_set = SRDataset(args.train_data, transform=transform)
    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    recorder = CSVwriter(os.path.join(args.out, "record.csv"))
    recorder.register(["Epoch", "LR", "Loss", "Train", "Valid"])

    best_psnr = 0
    res = []

    for epoch in range(args.total_step//args.valid_step):
        model.sample("max")
        loss, train_psnr = train(epoch, args.valid_step, model, train_loader, criterion, optimizer, scheduler)
        model.sample("max")
        psnr = valid(model, valid_loader, 128)
        lr = scheduler.get_last_lr()[0]
        recorder.update([epoch, round(lr, 6), round(loss, 4), round(train_psnr, 3), round(psnr, 3)])
        res.append(psnr)
        print([epoch, round(lr, 6), round(loss, 4), round(train_psnr, 3), round(psnr, 3)])

        wandb.log({"Train/Loss": loss}, step=epoch)
        wandb.log({"Train/PSNR": train_psnr}, step=epoch)
        wandb.log({"Test/PSNR": psnr}, step=epoch)
        wandb.log({"LR": lr}, step=epoch)

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
        torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(range(len(res)), res, label=f"depth={args.layers}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(args.out, "psnr.png"))
        plt.close()
    
    print(epoch, psnr, best_psnr)
    psnr = valid(model, valid_loader, None)
    print(psnr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, required=True, help='exp suffix')
    parser.add_argument('--train_data', type=str, required=False, default="/root/rjchen/data/SR/DIV2K_train_2.h5", help='train data folder')
    parser.add_argument('--valid_data', type=str, required=False, default="/root/rjchen/data/SR/DIV2K_valid_2.h5", help='valid data folder')
    parser.add_argument('--pretrained', type=str, required=False, default=None, help='pretrained weights')
    parser.add_argument('--out', type=str, default="../outputs/SwinIR", help='output folder')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_feat', type=int, default=32)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--total_step', type=int, default=64*2**10)
    parser.add_argument('--valid_step', type=int, default=2**10)
    parser.add_argument('--upsample', action="store_true")
    args = parser.parse_args()
    args.out = os.path.join(args.out, f"{args.suffix}_X{args.scale}")
    os.makedirs(args.out, exist_ok=True)

    main(args)

"""
CUDA_VISIBLE_DEVICES=6 python train.py --suffix full_min --out ../outputs/SwinIR3dev
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