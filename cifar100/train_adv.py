import argparse
import logging
import sys
import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision

from preactresnet import PreActResNet18

from utils import *
from attack_utils import attack_pgd, attack_pgd_eval


upper_limit, lower_limit = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chkpt-iters', default=50, type=int)
    parser.add_argument('--trial', default=2, type=int, help='how many trails to run')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    # Setup log file
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    permute_index = np.split(np.random.permutation(len(trainset)), args.trial)
    np.save('./{}/permute_index.npy'.format(args.fname), permute_index)

    # Setup learning rate schedule
    def lr_schedule(t):
        if t / args.epochs < 0.5:
            return args.lr_max
        elif t / args.epochs < 0.75:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.

    for trial in range(args.trial):

        # Setup training
        print('trial: ', trial)
        ## Set up train subset ##
        trainsubset = get_subsample_dataset(trainset, permute_index[trial])
        trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        # Specify attack eps/step-size
        epsilon = (args.epsilon / 255.)
        pgd_alpha = (args.epsilon / (255. * 4.))
        print('pgd_alpha:', pgd_alpha)

        # Setup model
        if args.model == 'PreActResNet18':
            model = PreActResNet18(num_classes=100)
        else:
            raise ValueError("Unknown model")
        model = nn.DataParallel(model).cuda()

        # Save init model
        torch.save(model.state_dict(), os.path.join(args.fname, f'model_{trial}_init.pth'))

        # Setup optimizer
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        best_test_robust_acc = 0
        start_epoch = 0

        logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
        for epoch in range(start_epoch, args.epochs):

            # Learning rate schedule
            lr = lr_schedule(epoch)
            opt.param_groups[0].update(lr=lr)

            ##################################################################
            # Training
            ##################################################################
            model.train()
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_robust_loss = 0
            train_robust_acc = 0
            train_n = 0
            for i, (inputs, targets) in enumerate(trainloader):

                X, y = inputs.cuda(), targets.cuda()

                if args.attack == 'pgd':
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.norm)
                    delta = delta.detach()
                # Standard training
                elif args.attack == 'none':
                    delta = torch.zeros_like(X)

                robust_output = model(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
                robust_loss = criterion(robust_output, y)

                opt.zero_grad()
                robust_loss.backward()
                opt.step()

                output = model(X)
                loss = criterion(output, y)

                train_robust_loss += robust_loss.item() * y.size(0)
                train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)

            train_time = time.time()

            ##################################################################
            # Evaluation
            ##################################################################
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            for i, (inputs, targets) in enumerate(testloader):
                X, y = inputs.cuda(), targets.cuda()

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd_eval(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.norm)
                delta = delta.detach()

                robust_output = model(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
                robust_loss = criterion(robust_output, y)

                output = model(X)
                loss = criterion(output, y)

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            test_time = time.time()

            ##################################################################
            # Log
            ##################################################################
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            ##################################################################
            # Save checkpoints
            ##################################################################
            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == args.epochs or epoch in [99, 100, 101, 149, 150, 151]:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{trial}_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{trial}_{epoch}.pth'))
            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_{trial}_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n


if __name__ == "__main__":
    main()
