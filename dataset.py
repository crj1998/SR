
import h5py
import random

from PIL import Image

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, *imgs):
        if random.random() < self.p:
            return tuple(TF.hflip(img) for img in imgs)
        return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(nn.Module):
    def __init__(
        self, degrees
    ):
        super().__init__()
        self.degrees = degrees

    def forward(self, *imgs):
        angle = random.choice(self.degrees)
        return tuple(TF.rotate(img, angle) for img in imgs)

    def __repr__(self) -> str:
        f"{self.__class__.__name__}(degrees={self.degrees})"


class ToTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *imgs):
        return tuple(TF.to_tensor(img) for img in imgs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class SRDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.lr = None
        self.hr = None
        with h5py.File(path, 'r') as f:
            self.len = len(f["lr"])
 
    def __getitem__(self, index):
        if self.lr is None or self.hr is None:
            f = h5py.File(self.path, 'r')
            self.lr, self.hr = f["lr"], f["hr"]

        lr, hr = self.lr[index], self.hr[index]
        lr, hr = Image.fromarray(lr), Image.fromarray(hr)
        if self.transform is not None:
            if isinstance(self.transform, list):
                for transform in self.transform:
                    lr, hr = transform(lr, hr)
            else:
                lr, hr = self.transform(lr), self.transform(hr)
        return lr, hr
 
    def __len__(self):
        return self.len


if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image
    transform = [
        RandomRotation([0, 90, 180, 270]), 
        RandomHorizontalFlip(0.5), 
        ToTensor()
    ]
    dataset = SRDataset("/root/rjchen/data/SR/DIV2K_valid_2.h5", transform)
    print(len(dataset))
    print(dataset[1][0].shape, dataset[1][1].shape)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    lr, hr = next(iter(dataloader))

    
    save_image(make_grid(lr, nrow=4), "lr.png")
    save_image(make_grid(hr, nrow=4), "hr.png")


