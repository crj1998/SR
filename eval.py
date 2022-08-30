


from tqdm import tqdm
from itertools import product

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from model import FSRCNN
# from supernet.fsrcnn import FSRCNN
# from model.swinir import SwinIR
from supernet.swinir import SwinIR

from dataset import SRDataset
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


def main(args):
    d, s, m = args.d, 12, 4
    state_dict = torch.load(args.weight)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if "qkv" in k:
    #         new_state_dict[k.replace("qkv", "q")] = v[:v.size(0)//3].clone()
    #         new_state_dict[k.replace("qkv", "k")] = v[v.size(0)//3:2*v.size(0)//3].clone()
    #         new_state_dict[k.replace("qkv", "v")] = v[2*v.size(0)//3:].clone()
    #     elif "upsample.0" in k:
    #         new_state_dict[k.replace("0", "conv1")] = v.clone()
    #     elif "upsample.2.0" in k:
    #         new_state_dict[k.replace("2.0", "expand")] = v.clone()
    #     elif "upsample.3" in k:
    #         new_state_dict[k.replace("3", "conv2")] = v.clone()
    #     else:
    #         new_state_dict[k] = v.clone()
    # state_dict = {k[len("module."):]: v.clone() for k, v in state_dict.items()}

    # model = FSRCNN(args.scale, 3, d, s, m)
    # model = SwinIR(
    #     upscale=args.scale, img_size=(64, 64), patch_size=1, embed_dim=64, num_feat=64,
    #     window_size=4, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], mlp_ratio=2.0
    # )
    model = SwinIR(
        upscale=args.scale, img_size=(64, 64), patch_size=1, embed_dim=64, num_feat=64,
        window_size=4, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], mlp_ratio=2.0
    )

    model.load_state_dict(state_dict)


    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    candidates = list(product([16, 32, 64], [16, 32, 64]))
    for cand in candidates:
        if hasattr(model, "set_sample_config"):
            model.set_sample_config(*cand)
        model = model.to(device)
        psnr = valid(model, valid_loader)

        print(cand, psnr)




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
CUDA_VISIBLE_DEVICES=5 python eval.py --valid_data ../../data/SR/DIV2K_valid_2.h5 --weight ../outputs/swinir_supernet_ddp_X2/best.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --valid_data ../../data/SR/DIV2K_valid_2.h5 --weight ../outputs/swinir_supernet_ddp_fixed_finetune_X2/best.pth
python eval.py --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --weight /root/rjchen/workspace/outputs/swinir_supernet_X2/best.pth

"""