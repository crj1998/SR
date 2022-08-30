import os, sys, glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch



from model import FSRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: dim check
rgb2y = lambda x: 16./255. + (64.738 / 256. * x[..., 0] + 129.057 / 256. * x[..., 1] + 25.064 / 256. * x[..., 2])

def calc_psnr(img1, img2):
    y1, y2 = rgb2y(img1), rgb2y(img2)
    return 10 * np.log10(1 / max(((y1-y2)**2).mean(), 1e-8))

save = lambda x, f: Image.fromarray((x*255).astype(np.uint8)).save(f)


def setup_model(arch, weight):
    d, s, m = 56, 12, 4
    scale = 2
    model = FSRCNN(scale, 3, d, s, m)

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model

@torch.no_grad()
def patch_inference(model, image, size, stride, scale):
    h, w, c = image.shape
    H, W = scale*h, scale*w

    # Decomposition: image -> patch
    ij = np.stack(np.meshgrid(np.arange(0, h-size+1, stride), np.arange(0, w-size+1, stride)), -1).reshape(-1, 2)

    i, j = ij.T
    x = np.arange(size)
    y = np.arange(size)
    z = np.arange(c)
    patches = np.take(image, x[:, None, None]*w*c + y[:, None]*c + z + (i*w*c + j*c)[:, None, None, None])

    # patch super resolution
    patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2).to(device)
    patches = model(patches).clamp(0, 1)
    patches = patches.cpu().permute(0, 2, 3, 1).numpy()

    # Combination: patch -> image

    sr = np.zeros((H, W, c)) - 1

    for t, (i, j) in enumerate(ij):
        patch = sr[i*scale:(i+size)*scale, j*scale:(j+size)*scale]
        overlap = (patch != -1)
        patch[overlap] = (patch[overlap] + patches[t][overlap]) / 2
        patch[~overlap] = patches[t][~overlap]

    return sr

@torch.no_grad()
def image_inference(model, image, size, stride, scale):
    # patch super resolution
    sr = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(dim=0).to(device)
    sr = model(sr).clamp(0, 1)
    sr = sr.squeeze(dim=0).cpu().permute(1, 2, 0).numpy()

    return sr

def main(root, size, stride, scale, weight):
    model = setup_model("fsrcnn", weight).to(device)
    PSNRs = []

    with tqdm(glob.glob(f"{root}/*.png")) as t:
        for f in t:
            im = Image.open(f).convert("RGB")
            W, H = im.size

            m = (H - scale*(size-stride))//(scale*stride)
            n = (W - scale*(size-stride))//(scale*stride)

            w, h = stride * n + size - stride, stride * m + size - stride
            w, h = w * scale, h * scale

            hr = im.crop(((W-w)//2, (H-h)//2, (W-w)//2 + w, (H-h)//2 + h))
            lr = im.resize((w//scale, h//scale), resample=Image.Resampling.BICUBIC)

            lr, hr = np.asarray(lr)/255., np.asarray(hr)/255.

            sr = image_inference(model, lr, size, stride, scale)

            psnr = calc_psnr(sr, hr)

            PSNRs.append(psnr)

            t.set_postfix_str(f"{psnr:.2f} dB @{os.path.basename(f)}")

            if psnr == np.max(PSNRs):
                save(lr, "lr.png")
                save(hr, "hr.png")
                save(sr, "sr.png")
    
    print(np.mean(PSNRs))
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help='model weight')
    parser.add_argument('--root', type=str, required=True, help='data folder')
    args = parser.parse_args()

    main(args.root, 64, 48, 2, args.weight)

"""
python inference.py --root /root/rjchen/data/SR/test2k/HR/X4 --weight /root/rjchen/workspace/outputs/base_l1_X2/best.pth
"""