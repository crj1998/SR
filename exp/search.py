import sys, os
sys.path.append("/root/rjchen/workspace/SR")

import csv
from tqdm import tqdm
from thop import profile

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from model import FSRCNN
# from supernet.fsrcnn import FSRCNN
# from model.swinir import SwinIR
# from supernet.swinir import SwinIR
# from supernet.swinir2 import SwinIR
from supernet.swinir3 import SwinIR

from dataset import SRDataset
from utils import AverageMeter, batch_psnr

DISABLE = False



@torch.no_grad()
def valid(model, dataloader):
    PSNR = AverageMeter()
    model.eval()
    with tqdm(dataloader, total=len(dataloader), desc="Valid", ncols=100, disable=DISABLE) as t:
        for lr, hr in t:
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            psnr = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr)
            PSNR.update(psnr.item(), batch_size)
            t.set_postfix({"PSNR": f"{PSNR.item():.2f} dB"})
    return PSNR.item()


def main(args):
    d, s, m = args.d, 12, 4
    # model = FSRCNN(args.scale, 3, d, s, m)
    # model = SwinIR(
    #     upscale=args.scale, img_size=(64, 64), patch_size=1, embed_dim=64, num_feat=64,
    #     window_size=4, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], mlp_ratio=2.0
    # )
    # swinir1
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
    model = SwinIR(**init_kwargs)

    # [4]*4, 38.23
    if isinstance(args.weight, str) and os.path.isfile(args.weight):
        model.load_state_dict(torch.load(args.weight))
    else:
        print(f"weight unload")
    model = model.to(device)

    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    inp = torch.rand((1, 3, 64, 64), device=device)

    PSNRs = []
    MACs = []
    cache = set()
    with open('subnets_sandwich.csv', 'w', newline='') as csvfile:
        fieldnames = ['layers', 'embed_dim', 'num_feat', 'num_heads_0', 'num_heads_1', 'num_heads_2', 'num_heads_3', 'mlp_ratio_0', 'mlp_ratio_1', 'mlp_ratio_2', 'mlp_ratio_3', 'psnr', 'macs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        cnt = 0
        for i in range(1024):
            if cnt > 512:
                break
            config = model.sample("random")
            fingerprint = ".".join(map(lambda x: str(x), [config['layers'], config['embed_dim'], config['num_feat']] + [config[f'num_heads_{i}'] for i in range(config['layers'])] +  [config[f'mlp_ratio_{i}'] for i in range(config['layers'])]))
            if fingerprint in cache:
                continue
            cnt += 1
            cache.add(fingerprint)
            psnr = valid(model, valid_loader)
            macs, params = profile(model, inputs=(inp, ), custom_ops=model.get_custom_ops(), verbose=False)
            PSNRs.append(psnr)
            MACs.append(macs/1e3)
            config.update({"psnr": round(psnr, 2), "macs": round(macs/1e6, 1)})
            writer.writerow(config)
            csvfile.flush()
            print(config, psnr, macs)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=160)
    # ax.scatter(MACs, PSNRs)
    # plt.savefig("out.png")
    # plt.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help='mdoel weight')
    parser.add_argument('--valid_data', type=str, default="/root/rjchen/data/SR/DIV2K_valid_2.h5", help='valid data folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=2 python search.py --weight /root/rjchen/workspace/outputs/SwinIR3dev/full_random_X2/best.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --valid_data ../../data/SR/DIV2K_valid_2.h5 --weight ../outputs/swinir_supernet_ddp_fixed_finetune_X2/best.pth
python eval.py --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --weight /root/rjchen/workspace/outputs/swinir_supernet_X2/best.pth

"""