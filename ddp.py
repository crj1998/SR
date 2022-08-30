import os, math, time, random
import argparse, logging
from tqdm import tqdm

import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from supernet.swinir import SwinIR
from dataset import SRDataset, RandomHorizontalFlip, RandomRotation, ToTensor
from utils import get_logger, colorstr, setup_seed, AverageMeter, batch_psnr

DISABLE = False

from contextlib import contextmanager

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

@contextmanager
def torch_distributed_zero_first(rank):
    """
    Decorator to make all processes in distributed training
    wait for each local_master to do something.
    """
    if rank not in [-1, 0]:
        dist.barrier(device_ids=[rank])
    yield
    if rank == 0:
        dist.barrier(device_ids=[0])


def all_gather(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)

def all_gather_object(obj):
    output_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output_objects, obj)
    return output_objects

@torch.no_grad()
def valid(model, dataloader, args):
    PSNR = AverageMeter()
    for lr, hr in dataloader:
        batch_size = lr.size(0)
        lr, hr = lr.to(args.device), hr.to(args.device)
        psnr = batch_psnr(model(lr).detach().clamp(0.0, 1.0), hr)
        PSNR.update(psnr.item(), batch_size)

    psnrs = all_gather(torch.tensor([PSNR.item()], device=args.device))
    psnr = psnrs.mean().item()
    if args.local_rank in [-1, 0]:
        args.logger.info(f"Valid: PSNR={psnr:.2f} dB")
    return psnr

def train(epoch, iters, model, dataloader, criterion, optimizer, scheduler, args):
    Loss = AverageMeter()
    PSNR = AverageMeter()

    dataiter = iter(dataloader)
    model.train()
    for it in range(iters):
        try:
            lr, hr = next(dataiter)
        except:
            dataiter = iter(dataloader)
            lr, hr = next(dataiter)
        if it % 5 == 1:
            unwrap_model(model).set_sample_config(64, 64)
        elif it % 5 == 3:
            unwrap_model(model).set_sample_config(16, 16)
        else:
            unwrap_model(model).set_sample_config(random.choice([16, 32, 64]), random.choice([16, 32, 64]))
        batch_size = lr.size(0)
        lr, hr = lr.to(args.device), hr.to(args.device)

        output = model(lr)
        loss = criterion(output, hr)
        psnr = batch_psnr(output.detach().clamp(0.0, 1.0), hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        Loss.update(loss.item(), batch_size)
        PSNR.update(psnr.item(), batch_size)

        # if it % 32 == 0:
        if args.local_rank in [-1, 0] and it % 128 == 0:
            lr = scheduler.get_last_lr()[0]
            args.logger.info(f"Iter {it:04d}: loss={Loss.item():.4f}, PSNR={PSNR.item():.2f} dB, LR={lr:.5f}")

    return Loss.item(), PSNR.item()


def main(args):

    # set up logger
    level = logging.DEBUG if "dev" in args.out else logging.INFO
    logger = get_logger(
        args = args,
        name = "SR",
        level = level if args.local_rank in [-1, 0] else logging.WARN,
        fmt = "%(asctime)s [%(levelname)s] %(message)s",
        rank = args.local_rank
    )
    args.logger = logger
    logger.debug(f"Get logger named {colorstr('SR')}!")
    logger.debug(f"distributed available? {dist.is_available()}")

    #setup random seed
    if args.seed and isinstance(args.seed, int):
        setup_seed(args.seed)
        logger.info(f"Setup random seed {colorstr('green', args.seed)}!")
    else:
        logger.info(f"Can not Setup random seed with seed is {colorstr('green', args.seed)}!")

    # init dist params
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        args.n_gpu = torch.cuda.device_count()
        args.local_rank = dist.get_rank()
        # torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        assert dist.is_initialized(), f"Distributed initialization failed!"

    # set device
    args.device = device
    logger.debug(f"Current device: {device}")

    # make dataset
    with torch_distributed_zero_first(args.local_rank):
        transform = [
            RandomRotation([0, 90, 180, 270]), 
            RandomHorizontalFlip(0.5), 
            ToTensor()
        ]
        train_set = SRDataset(args.train_data, transform=transform)
        valid_set = SRDataset(args.valid_data, transform=T.ToTensor())

        logger.info(f"Dataset: {len(train_set)} samples for train, {len(valid_set)} sampels for valid!")

    # make dataset loader
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    valid_sampler = SequentialSampler if args.local_rank == -1 else SequentialDistributedSampler

    # prepare labeled_trainloader
    train_loader = DataLoader(
        train_set,
        sampler = train_sampler(train_set),
        batch_size = args.batch_size//args.world_size,
        num_workers = args.num_workers,
        drop_last = True,
        pin_memory = True
    )

    # prepare valid_loader
    valid_loader = DataLoader(
        valid_set,
        sampler = valid_sampler(valid_set, args.batch_size//args.world_size),
        batch_size = args.batch_size//args.world_size,
        num_workers = args.num_workers,
        drop_last = True,
        pin_memory = True
    )

    logger.info(f"Dataloader Initialized. Batch size: {colorstr('green', args.batch_size)}, Num workers: {colorstr('green', args.num_workers)}.")

    with torch_distributed_zero_first(args.local_rank):
        # build model
        model = SwinIR(
            upscale=args.scale, img_size=(64, 64), patch_size=1, embed_dim=64, num_feat=64,
            window_size=4, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], mlp_ratio=2.0
        )
        logger.info(f"Model: {colorstr('SwinIR')}. Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # load from pre-trained, before DistributedDataParallel constructor
        # pretrained is str and it exists and is file.
        if isinstance(args.pretrained, str) and os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            logger.debug(f"Start load pretrained weights @: {args.pretrained}.")
            state_dict = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v.clone() for k, v in state_dict.items()}

            msg = model.load_state_dict(state_dict, strict=False)
            logger.warning(f"Missing keys {msg.missing_keys} in state dict.")

            logger.debug(f"Pretrained weights @: {args.pretrained} loaded!")

    model.to(args.device)
    criterion = nn.L1Loss(reduction='mean').to(args.device)
    # make optimizer, scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    warm_up = 0
    lr_min = 0.01
    T_max = args.total_step
    lr_lambda = lambda i: i / warm_up if i<warm_up else lr_min + (1-lr_min)*(1.0+math.cos((i-warm_up)/(T_max-warm_up)*math.pi))/2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(f"Optimizer {colorstr('Adam')} and Scheduler {colorstr('Cosine')} selected!")

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    model.zero_grad()
    best_psnr = 0
    
    #train loop
    for epoch in range(args.total_step//args.valid_step):
        train_loader.sampler.set_epoch(epoch)
        loss, _ = train(epoch, args.valid_step, model, train_loader, criterion, optimizer, scheduler, args)
        unwrap_model(model).set_sample_config(64, 64)
        psnr = valid(model, valid_loader, args)
        if args.local_rank in [-1, 0]:
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(unwrap_model(model).state_dict(), os.path.join(args.out, "best.pth"))
            torch.save(unwrap_model(model).state_dict(), os.path.join(args.out, "last.pth"))
            print(psnr, best_psnr)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--suffix', type=str, required=True, help='exp suffix')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--train_data', type=str, required=True, help='train data folder')
    parser.add_argument('--valid_data', type=str, required=True, help='valid data folder')
    parser.add_argument('--pretrained', default=None, help='directory to pretrained model')
    parser.add_argument('--out', type=str, required=True, help='output folder')
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--total_step', type=int, default=2**15)
    parser.add_argument('--valid_step', type=int, default=2**12)
    parser.add_argument('--upsample', action="store_true")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    
    args.out = os.path.join(args.out, f"{args.suffix}_X{args.scale}")
    os.makedirs(args.out, exist_ok=True)

    main(args)

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 ddp.py --suffix swinir_supernet_ddp_finetune --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs --pretrained ../outputs/swinir_supernet_X2/best.pth --lr 0.0005
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 29510 ddp.py --suffix swinir_supernet_ddp_fixed_finetune --train_data /root/rjchen/data/SR/DIV2K_train_2.h5 --valid_data /root/rjchen/data/SR/DIV2K_valid_2.h5 --out ../outputs  --pretrained ../outputs/swinir_small_X2/supernet.pth --lr 0.0005
"""