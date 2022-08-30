
import sys, os
sys.path.append("/root/rjchen/workspace/SR")

from tqdm import tqdm
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from supernet.swinir2 import SwinIR

from dataset import SRDataset
from utils import AverageMeter, batch_psnr

DISABLE = False

@torch.no_grad()
def valid(model, dataloader):
    PSNR = AverageMeter()
    psnrs = []
    with tqdm(dataloader, total=len(dataloader), desc="Valid", ncols=100, disable=DISABLE) as t:
        for lr, hr in t:
            batch_size = lr.size(0)
            lr, hr = lr.to(device), hr.to(device)
            psnr = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr, reduction="none")
            psnrs.append(psnr)
            PSNR.update(psnr.mean().item(), batch_size)
            t.set_postfix({"PSNR": f"{PSNR.item():.2f} dB"})
    psnrs = torch.cat(psnrs, dim=0)
    return psnrs

@torch.no_grad()
def valid_layer(model, dataloader):
    layer_diff = [[] for _ in range(4)]
    for lr, hr in tqdm(dataloader, total=len(dataloader), desc="valid_layer", ncols=100, disable=False):
        lr = lr.to(device)
        latent_features = model.forward_layer(lr)
        for i in range(1, len(latent_features)):
            # layer_diff[i-1].append((latent_features[i] - latent_features[i-1]).flatten(start_dim=1).abs().max(dim=-1).values)
            layer_diff[i-1].append((latent_features[i] - latent_features[i-1]).flatten(start_dim=1).mean(dim=-1).abs())

    layer_diff = [torch.cat(diff, dim=0) for diff in layer_diff]
    return layer_diff


FORCE_REDO = False

def main(args):
    state_dict = torch.load(args.weight)
    model = SwinIR(
        upscale=args.scale, embed_dim=args.embed_dim, num_feat=args.num_feat, 
        img_size=(64, 64), patch_size=1, window_size=4, 
        depths=[2]*4, num_heads=[4]*4, mlp_ratio=2.0
    )
    model.load_state_dict(state_dict)
    model = model.to(device)

    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    upsample = nn.Upsample(scale_factor=args.scale, mode="bilinear")  # "bicubic"
    if not FORCE_REDO and os.path.exists("upsample.npy"):
        scores = np.load("upsample.npy")
    else:
        scores = valid(upsample, valid_loader).cpu().numpy()
        np.save("upsample.npy", scores)
    print("Upsample", scores.mean())

    if not FORCE_REDO and os.path.exists(f"valid_layer.npy"):
        layer_diff = np.load(f"valid_layer.npy")
    else:
        layer_diff = [diff.cpu().numpy() for diff in valid_layer(model, valid_loader)]
        np.save(f"valid_layer.npy", layer_diff)

    # vmin, vmax = scores.min(), scores.max()
    # bins = 2 * np.arange(vmin//2, vmax//2+1, 1)
    # x = (bins[1:] + bins[:-1])/2
    # inds = np.digitize(scores, bins)

    # fig = plt.figure(figsize=(5, 5), dpi=160)
    # for idx, diff in enumerate(layer_diff):
    #     y = np.array([ diff[inds==ind].mean() if (inds==ind).sum() > 0 else 0 for ind in range(1, len(bins))])
    #     plt.scatter(x, y, label=f"d={idx+1}", s=7, alpha=0.9)
    #     # plt.scatter(scores, diff, label=f"d={idx+1}", s=5, alpha=0.7)
    # plt.title("Layer response difference")
    # plt.xlabel("Bilinear PSNR")
    # plt.ylabel("SwinIR PSNR")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("layer.png")
    # exit()


    candidates = [0, 1, 2, 3, 4]
    # candidates = [0,]
    # model.sample("random")
    # cand = model.sampled_layers
    PSNRS = {cand: None for cand in candidates}
    MACS = {cand: None for cand in candidates}
    PARAMS = {cand: None for cand in candidates}
    inp = torch.rand((1, 3, 64, 64), device=device)
    

    for cand in candidates:
        model.set_sample_config(cand)
        if not FORCE_REDO and os.path.exists(f"d={cand}.npy"):
            psnrs = np.load(f"d={cand}.npy")
        else:
            psnrs = valid(model, valid_loader).cpu().numpy()
            np.save(f"d={cand}.npy", psnrs)
        PSNRS[cand] = psnrs
        macs, params = profile(model, inputs=(inp, ), verbose=False)
        MACS[cand] = round(macs/1e6, 1)
        PARAMS[cand] = params
        print(f"SwinIR depth@{cand}: MACs={macs/1e6:9.2f} Params={params/1e3:6.2f} PSNR={psnrs.mean():.2f}dB")


    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(5, 5), dpi=160)

    ax = fig.add_axes(rect_scatter)
    ax_inset = fig.add_axes([0.15, 0.5, .2, .2], facecolor="white")
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax.set_xlabel("Bilinear PSNR")
    ax.set_ylabel("SwinIR PSNR")
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax_inset.set_title('Latency', {'fontsize': 8})
    ax_inset.set_xlabel("depth", fontsize=7)
    ax_inset.set_ylabel("MACs", fontsize=7, fontweight="bold")
    ax_inset.set(xticks=[], yticks=[])

    vmin, vmax = scores.min(), scores.max()
    bins = 2 * np.arange(vmin//2, vmax//2+1, 1)
    x = (bins[1:] + bins[:-1])/2
    inds = np.digitize(scores, bins)
    ax_histx.hist(scores, bins=len(bins), color="gray", alpha=0.5)
    ax.plot([vmin, vmax], [vmin, vmax], color="gray", linestyle="--")
    ax_inset.bar(candidates, [MACS[cand] for cand in candidates], color=[f"C{cand}" for cand in candidates], alpha=0.5)

    for cand in candidates:
        psnrs = PSNRS[cand]
        # vmin, vmax = min(vmin, psnrs.min()), max(vmax, psnrs.max())
        y = np.array([ psnrs[inds==ind].mean() if (inds==ind).sum() > 0 else 0 for ind in range(1, len(bins))])
        ax.scatter(x, y, label=f"d={cand}", s=5, alpha=0.7)
        
        ax_histy.hist(psnrs, bins=len(bins), orientation='horizontal', label=f"{cand}(+{round(psnrs.mean()-scores.mean(), 1):.1f})", alpha=0.5)

    # ax.grid()
    # (left+width+2*spacing, bottom+height+2*spacing)
    ax_histy.legend(bbox_to_anchor=(1.0, 1.33), prop={'size': 8}, markerscale=1.0, title="depth(+ dB)", borderpad=0.25, labelspacing=0.25)
    plt.savefig("output.png")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help='mdoel weight')
    parser.add_argument('--valid_data', type=str, default="/root/rjchen/data/SR/DIV2K_valid_2.h5", help='valid data folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_feat', type=int, default=32)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=6 python exp/valid.py --weight /root/rjchen/workspace/outputs/SwinIR2/depth4_dim32_X2/best.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --valid_data ../../data/SR/DIV2K_valid_2.h5 --weight ../outputs/swinir_supernet_ddp_fixed_finetune_X2/best.pth
python eval.py --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --weight /root/rjchen/workspace/outputs/swinir_supernet_X2/best.pth

"""