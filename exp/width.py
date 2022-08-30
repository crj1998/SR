
import sys
sys.path.append("/root/rjchen/workspace/SR")

import pickle
from tqdm import tqdm
import numpy as np


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from model import FSRCNN
from supernet.fsrcnn import FSRCNN
from dataset import SRDataset
from utils import batch_psnr

DISABLE = False

@torch.no_grad()
def valid(model, dataloader, desc):
    PSNRs = []
    for lr, hr in tqdm(dataloader, total=len(dataloader), desc=desc, ncols=100, disable=DISABLE):
        lr, hr = lr.to(device), hr.to(device)
        psnrs = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr, False, "none").cpu().numpy()
        PSNRs.append(psnrs)
    PSNRs = np.concatenate(PSNRs, axis=0)
    return PSNRs


# def main(args):
#     valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
#     valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

#     baseline = torch.nn.Upsample(scale_factor=args.scale, mode="bicubic", align_corners=False)
#     baseline = baseline.to(device)
#     psnr_baseline = valid(baseline, valid_loader, "Baeline")

#     data = {"baseline": psnr_baseline.copy()}

#     for d in [7, 14, 28, 56, 112]:
#         if args.upsample:
#             weight = f"/root/rjchen/workspace/outputs/fsrcnn_l1_d{d}_upsample_X2/best.pth"
#         else:
#             weight = f"/root/rjchen/workspace/outputs/fsrcnn_l1_d{d}_X2/best.pth"
#         model = FSRCNN(args.scale, 3, d, 12, 4, args.upsample)
#         model.load_state_dict(torch.load(weight))
#         model = model.to(device)

#         if hasattr(model, "set_sample_config"):
#             model.set_sample_config(d)
        
#         psnr = valid(model, valid_loader, f"d={d}")
        
#         data[str(d)] = psnr.copy()
    
#     with open("/root/rjchen/workspace/outputs/cache/width.pkl", "wb") as f:
#         pickle.dump(data, f)

    

def main(args):
    valid_set = SRDataset(args.valid_data, transform=T.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = FSRCNN(args.scale, 3, args.d, 12, 4, args.upsample)
    if args.upsample:
        weight = f"/root/rjchen/workspace/outputs/fsrcnn_l1_supernet_upsample_X2/best.pth"
    else:
        weight = f"/root/rjchen/workspace/outputs/fsrcnn_l1_supernet_X2/best.pth"
    model.load_state_dict(torch.load(weight))
    model = model.to(device)

    baseline = torch.nn.Upsample(scale_factor=args.scale, mode="bicubic", align_corners=False)
    baseline = baseline.to(device)
    psnr_baseline = valid(baseline, valid_loader, "Baeline")

    data = {"baseline": psnr_baseline.copy()}


    for d in range(7, 57, 7):
        # if hasattr(model, "set_sample_config"):
        #     model.set_sample_config(d)
        model.set_sample_config(d)

        psnr = valid(model, valid_loader, f"d={d}")
        
        data[str(d)] = psnr.copy()
    
    with open("/root/rjchen/workspace/outputs/cache/super_width_upsample.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_data', type=str, default="/root/rjchen/data/SR/DIV2K_valid_2.h5", help='valid data folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--upsample', action="store_true")
    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=4 python exp/width.py
CUDA_VISIBLE_DEVICES=4 python exp/width.py --upsample
"""