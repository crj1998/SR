import h5py
from PIL import Image
from tqdm import tqdm

import numpy as np


# TODO: dim check
rgb2y = lambda x: 16./255. + (64.738 / 256. * x[..., 0] + 129.057 / 256. * x[..., 1] + 25.064 / 256. * x[..., 2])

def calc_psnr(img1, img2):
    y1, y2 = rgb2y(img1), rgb2y(img2)
    return 10 * np.log10(1 / max(((y1-y2)**2).mean(), 1e-8))


h5_file = "/root/rjchen/data/SR/DIV2K_train_2.h5"
h5_file = h5py.File(h5_file, 'r')

PSNRs = []
for lr, hr in tqdm(zip(h5_file["lr"], h5_file["hr"]), desc="Processing..."):
    H, W, C = hr.shape
    h, w, c = lr.shape
    lr = Image.fromarray(lr)
    lr = lr.resize((W, H), resample=Image.Resampling.BICUBIC)
    
    lr = np.asarray(lr)/255.
    hr = hr/255.

    psnr = calc_psnr(lr, hr)
    PSNRs.append(psnr)
np.save("../outputs/cache/train_bicubic_psnr.npy", np.array(PSNRs))
print(np.mean(PSNRs))



h5_file.close()