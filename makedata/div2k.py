import os, sys
import glob, h5py

from itertools import product

from PIL import Image
from tqdm import tqdm

import numpy as np


floor_int = lambda x, y: int((x // y) * y)


"""
PNG: 1 pixel = 32 bit = 4 bytes
"""
def patchify(im, size, stride):
    W, H = im.size
    im = np.asarray(im, dtype=np.uint8)

    for i, j in product(
        range(0, H - size + 1, stride), 
        range(0, W - size + 1, stride)
    ):
        yield im[i:i+size, j:j+size]



def main(root, output, scale=2, size=64, stride=32, aug=True):
    patches = 0
    storage = 0
    hr_patches = []
    lr_patches = []
    scales = [1.0, 0.8, 0.6] if aug else [1.0]
    with tqdm(glob.glob(f'{root}/*.png')[::4], desc="Processing...", ncols=120) as t:
        for f in t:
            im = Image.open(f).convert("RGB")
            W, H = im.size
            for s in scales:
                # mod width height into  lowest common multiple of size and scale 
                w, h = floor_int(int(W*s), int(np.lcm(size, scale))), floor_int(int(H*s), int(np.lcm(size, scale)))
                # high resolution and low resolution
                hr = im.crop(((W - w)//2, (H - h)//2, (W - w)//2 + w, (H - h)//2 + h))
                lr = hr.resize((w//scale, h//scale) , resample=Image.Resampling.BICUBIC)

                # patchify large image
                for lr_patch, hr_patch in zip(patchify(lr, size, stride), patchify(hr, size*scale, stride*scale)):
                    lr_patches.append(lr_patch)
                    hr_patches.append(hr_patch)
                    patches += 2
                    storage += lr_patch.nbytes + hr_patch.nbytes
            t.set_postfix_str(f"@{os.path.basename(f)}. p: {patches/10**3:.1f} k, s: {storage/2**30:.2f} GB")

    lr_patches = np.stack(lr_patches, axis=0)
    hr_patches = np.stack(hr_patches, axis=0)
    print(f"dtype: {lr_patches.dtype} \n  LR: {lr_patches.nbytes/2**30:.2f} GB, {lr_patches.shape} \n  HR: {hr_patches.nbytes/2**30:.2f} GB, {hr_patches.shape}")

    h5_file = h5py.File(output, 'w')
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='data folder')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patchsize', type=int, default=64)
    parser.add_argument('--aug', action='store_true')
    args = parser.parse_args()
    if "vlaid" in args.root and args.aug:
        assert False, f"Valid but auged"
    main(args.root, args.output, args.scale, args.patchsize, args.patchsize//2, args.aug)

"""
python makedata/div2k.py --root /root/rjchen/data/SR/DIV2K_valid_HR --output /root/rjchen/data/SR/DIV2K_valid_2.h5 --scale 2
python makedata/div2k.py --root /root/rjchen/data/SR/DIV2K_train_HR --output /root/rjchen/data/SR/DIV2K_train_2.h5 --scale 2 --aug
python makedata/div2k.py --root /root/rjchen/data/SR/DIV2K_valid_HR --output /root/rjchen/data/SR/DIV2K_valid_3.h5 --scale 3
python makedata/div2k.py --root /root/rjchen/data/SR/DIV2K_train_HR --output /root/rjchen/data/SR/DIV2K_train_3.h5 --scale 3 --aug
python makedata/div2k.py --root /root/rjchen/data/SR/DIV2K_valid_HR --output /root/rjchen/data/SR/DIV2K_valid_4.h5 --scale 4
python makedata/div2k.py --root /root/rjchen/data/SR/DIV2K_train_HR --output /root/rjchen/data/SR/DIV2K_train_4.h5 --scale 4 --aug
"""