import argparse
import shutil
import time

import torch

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os 

import torch.multiprocessing as mp
import torch.distributed as dist

import math


from torchvision.datasets import CIFAR10
from Models.ResNetTorch import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
import wandb

import socket
from contextlib import closing


try:
    import torch.cuda.amp as amp
except ImportError:
    raise ImportError("Your version of PyTorch is too old.")

best_prec1 = 0


def parse():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument(
        "-data",
        "--data",
        default="ML/",
        type=str,
        metavar="DIR",
        help="path to dataset",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--epochs",
        default=180,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 128)",
    )

    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument("--local_rank", default=0, type=int)

    # My additional args
    parser.add_argument("--model", type=str, default="ResNet20")
    parser.add_argument("--CIFAR10", type=bool, default=False)
    parser.add_argument("--Mixed-Precision", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--step-lr", type=bool, default=True)
    parser.add_argument("--base-lr", type=float, default=0.1)


    #Distributed training utils
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    args = parser.parse_args()
    return args


def main():

    global best_prec1, args

    args = parse()

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def cleanup():
    dist.destroy_process_group()

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu


        free_port = get_open_port()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(free_port)

        dist.init_process_group(backend=args.dist_backend,
                                world_size=args.world_size, rank=args.rank)



    cudnn.benchmark = True

    if torch.cuda.is_available():

        if args.model == "ResNet20":
            model = ResNet20()

        elif args.model == "ResNet32":
            model = ResNet32()

        elif args.model == "ResNet44":
            model = ResNet44()

        elif args.model == "ResNet56":
            model = ResNet56()

        elif args.model == "ResNet110":
            model = ResNet110()


        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)


        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.cos_anneal:
        assert args.step_lr == False

        optimizer = create_optimizer(model, args.weight_decay, args.base_lr)

    if args.step_lr:
        assert args.cos_anneal == False

        optimizer = create_optimizer(model, args.weight_decay, args.base_lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[90, 130], gamma=0.1
        )

    if args.CIFAR10:
        assert args.num_classes == 10, "Must have 10 output classes for CIFAR10"
        # Use CIFAR-10 data augmentations
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(
                    (32, 32),
                    padding=4,
                    fill=0,
                    padding_mode="constant",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

        train_dataset = CIFAR10(
            root="./CIFAR", train=True, download=True, transform=transform_train
        )

        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [45000, 5000]
        )

        test_dataset = CIFAR10(
            root="./CIFAR", train=False, download=True, transform=transform_test
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Setup WandB logging here
    wandb_run = wandb.init(project="Flax Torch")
    wandb.config.max_epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.weight_decay = args.weight_decay

    wandb.config.ModelName = args.model
    wandb.config.Dataset = "CIFAR10"
    wandb.config.Package = "PyTorch"

    scaler = None
    if args.Mixed_Precision:
        scaler = amp.GradScaler()

    for epoch in range(0, args.epochs):

        if args.cos_anneal:
            lr = adjust_learning_rate(optimizer, epoch, args)
            scheduler = None

        train_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scaler=scaler,
            scheduler=scheduler,
        )
        if args.step_lr:
            scheduler.step()
            lr = (scheduler.get_last_lr())[0]

        _, _, val_loss = validate(validation_loader, model, criterion)

        if epoch % 10 == 0:

            prec1, prec5, test_loss = validate(test_loader, model, criterion)
            wandb.log(
                {
                    "acc@1": prec1,
                    "Learning Rate": lr,
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                }
            )

        else:
            wandb.log(
                {
                    "Learning Rate": lr,
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                }
            )

        # remember best prec@1 and save checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        if scaler is not None:
            with amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Measure accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.item()

        # to_python_float incurs a host<->device sync
        losses.update((reduced_loss), images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        if i % args.print_freq == 0 and args.local_rank == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Loss {losses.test:.10f} ({losses.avg:.4f})\t Prec@1 {top1.test.item():.3f} ({top1.avg.item():.3f})Prec@5 {top5.test.item():.3f} ({top5.avg.item():.3f})"
            )

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.test = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, test, n=1):
        self.test = test
        self.sum += test * n
        self.count += n
        self.avg = self.sum / self.count


def validate(loader, model, criterion, scaler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (images, target) in enumerate(loader):
        # compute output
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.item()

        losses.update((reduced_loss), images.size(0))
        top1.update((prec1), images.size(0))
        top5.update((prec5), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f"* Prec@1 {top1.avg.item():.3f} Prec@5 {top5.avg.item():.3f}")

    return top1.avg.item(), top5.avg.item(), losses.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified testues of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.base_lr
    if hasattr(args, "warmup") and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup))
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # for tracking
    return lr


def create_optimizer(model, weight_decay, lr):
    params = []
    for key, value in model.named_parameters():

        if "fc.bias" in key or "bias" in key or "bn" in key:
            print(f"No weight decay for paramater: {key}")
            apply_weight_decay = 0
            params += [
                {"params": [value], "lr": lr, "weight_decay": apply_weight_decay}
            ]

        else:
            apply_weight_decay = weight_decay
            params += [
                {"params": [value], "lr": lr, "weight_decay": apply_weight_decay}
            ]

    return torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)


if __name__ == "__main__":
    main()
    cleanup()
