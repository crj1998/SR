import logging
import os, sys, time, csv
import random
import numpy as np
from datetime import datetime

import torch

def colorstr(*inputs):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])  # color arguments, string
    string = str(string)
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def get_logger(args, name, level=logging.INFO, fmt="%(asctime)s - [%(levelname)s] %(message)s", rank=""):
    logger = logging.getLogger(name)
    # unlike the root logger, a custom logger can’t be configured using basicConfig()
    logging.basicConfig(
        filename = os.path.join(args.out, f"{time_str() if level==logging.INFO else 'dev'}_{rank}.log"),
        format = fmt, datefmt="%Y-%m-%d %H:%M:%S", level = level
    )
    logger.setLevel(level)

    # console print
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

    return logger


def setup_seed(seed):
    """ set seed for the whole program for removing randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)

@torch.no_grad()
def batch_rgb2y(imgs):
    assert imgs.ndim == 4
    return 16./255. + (64.738 / 256. * imgs[:, 0, ...] + 129.057 / 256. * imgs[:, 1, ...] + 25.064 / 256. * imgs[:, 2, ...])

@torch.no_grad()
def batch_psnr(imgs1, imgs2, rgb=False, reduction="mean"):
    assert isinstance(imgs1, torch.Tensor) and isinstance(imgs2, torch.Tensor)
    if rgb:
        mse = ((imgs1 - imgs2)**2).mean(dim=(-3, -2, -1))
    else:
        y1, y2 = batch_rgb2y(imgs1), batch_rgb2y(imgs2)
        mse = ((y1 - y2)**2).mean(dim=(-2, -1))
    if reduction == "none":
        return 10. * torch.log10(1. / torch.clamp(mse, 1e-8))
    else:
        return 10. * torch.log10(1. / torch.clamp(mse, 1e-8)).mean(dim=0)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, val, num=1):
        self.total += val * num
        self.count += num

    def item(self):
        return self.total/self.count

class CSVwriter:
    def __init__(self, filename, head=None):
        self.filename = filename
        self.head = head
        if head is not None:
            self.register(head)
        
    def register(self, head):
        self.head = head

        with open(self.filename, mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.head and isinstance(head, (list, tuple)):
                writer.writerow(["Time", *self.head])
            else:
                pass
    
    def update(self, row):
        with open(self.filename, mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            if isinstance(row, list):
                writer.writerow([datetime.today().strftime('%Y-%m-%d %H:%M:%S'), *row])
            elif isinstance(row, dict):
                writer.writerow([datetime.today().strftime('%Y-%m-%d %H:%M:%S'), *(row.get(k, "None") for k in self.head)])
            else:
                pass