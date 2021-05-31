import argparse
# import torchvision.models as models
import os

import numpy as np
import torch

import models


def arg_parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch COVID Training')
    parser.add_argument('--data-path', metavar='DIR',default='./datasets/china.csv',
                        help='path to dataset')
    parser.add_argument('--model', metavar='ARCH', default='cnn',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4)')
    parser.add_argument('--learning-rate', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-t', '--test', dest='test', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    return parser


def adjust_learning_rate(optimizer, epoch, args, warm=10):
    if epoch < warm:
        lr = args.learning_rate * (epoch / warm)
    else:
        epoch = epoch - warm
        total = args.epochs - warm
        factor = (np.cos(epoch*3.1415926/total)+1)/2
        lr = args.learning_rate * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # print(lr)


def save_weight(model, args, epoch):
    state = model.state_dict()
    dir_name = os.path.join("./weights/", args.model)
    os.makedirs(dir_name, exist_ok=True)
    path_name = os.path.join(dir_name, f"epoch{epoch+1}.pth")
    torch.save(state, path_name)


def load_weight(model, args):
    path_name = args.resume
    print(f"Loading Weight From '{path_name}'...")
    model.load_state_dict(torch.load(path_name))
    print(f"Weight Loaded.")
