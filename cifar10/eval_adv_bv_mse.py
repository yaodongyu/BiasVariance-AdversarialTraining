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

from wideresnet import WideResNet

from utils import *
import copy
from attack_utils import attack_pgd_bv_eval


upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='WideResNet')
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=20, type=int)
    parser.add_argument('--pgd-alpha', default=1, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--loadbest', action='store_true')
    parser.add_argument('--trial', default=2, type=int, help='how many trails to run')
    parser.add_argument('--test-size', default=10000, type=int, help='number of test points')
    return parser.parse_args()

args = get_args()
# loss definition
criterion_mse = nn.MSELoss(reduction='mean').cuda()
# variables for bias-variance calculation
NUM_CLASSES = 10
OUTPUST_SUM = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
OUTPUTS_SUMNORMSQUARED = torch.Tensor(args.test_size).zero_().cuda()

def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.long().cuda()
        targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)

        # attack
        X, y = inputs.cuda(), targets.cuda()
        if args.attack == 'none':
            inputs_adv = torch.clamp(X, min=lower_limit, max=upper_limit)
        else:
            epsilon = (args.epsilon / 255.)
            pgd_alpha = (3. * args.epsilon / (255. * args.attack_iters))
            delta = attack_pgd_bv_eval(net, X, y, epsilon, pgd_alpha, args.attack_iters, args.norm)
            delta = delta.detach()
            inputs_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)

        outputs = net(inputs_adv)
        outputs = outputs.detach()

        outputs = F.softmax(outputs, dim=1)
        loss = criterion_mse(outputs, targets_onehot)
        test_loss += loss.item() * outputs.numel()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    print('accuracy: ', 100. * correct / total)
    return test_loss / total, 100. * correct / total

def compute_bias_variance(net, testloader, trial):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)

        # attack
        X, y = inputs.cuda(), targets.cuda()
        if args.attack == 'none':
            inputs_adv = torch.clamp(X, min=lower_limit, max=upper_limit)
        else:
            epsilon = (args.epsilon / 255.)
            pgd_alpha = (3. * args.epsilon / (255. * args.attack_iters))
            delta = attack_pgd_bv_eval(net, X, y, epsilon, pgd_alpha, args.attack_iters, args.norm)
            delta = delta.detach()
            inputs_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)

        outputs = net(inputs_adv)
        outputs = outputs.detach()
        outputs = F.softmax(outputs, dim=1)

        OUTPUST_SUM[total:(total + targets.size(0)), :] += outputs
        OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)] += outputs.norm(dim=1) ** 2.0

        bias2 += (OUTPUST_SUM[total:total + targets.size(0), :] / (trial + 1) - targets_onehot).norm() ** 2.0
        variance += OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)].sum() / (trial + 1) - (OUTPUST_SUM[total:total + targets.size(0), :] / (trial + 1)).norm() ** 2.0
        total += targets.size(0)

    return bias2 / total, variance / total


##################################################
# setup log file
##################################################
if args.attack == 'none':
    epsilon_file = 0
else:
    epsilon_file = args.epsilon
if args.loadbest:
    logfilename = os.path.join(args.fname, 'bv_mse_log_best_eps{}.txt'.format(epsilon_file))
else:
    logfilename = os.path.join(args.fname, 'bv_mse_log_epoch{}_eps{}.txt'.format(args.resume, epsilon_file))
init_logfile(logfilename, "trial\ttest loss\ttest acc\tbias2\tvariance")
# train/test accuracy/loss
TEST_ACC_SUM = 0.0
TEST_LOSS_SUM = 0.0

# Setup test loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if not os.path.exists(args.fname):
    os.makedirs(args.fname)

for trial in range(args.trial):

    # Setup model
    if args.model == 'WideResNet':
        model = WideResNet(28, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")
    model = nn.DataParallel(model).cuda()
    model.eval()

    # Load model checkpoint
    start_epoch = args.resume
    if args.loadbest:
        print('loading model: ', os.path.join(args.fname, f'model_{trial}_best.pth'))
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{trial}_best.pth'))['state_dict'])
    else:
        if start_epoch == 0:
            print('loading model: ', os.path.join(args.fname, f'model_{trial}_init.pth'))
            model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{trial}_init.pth')))
        else:
            print('loading model: ', os.path.join(args.fname, f'model_{trial}_{start_epoch - 1}.pth'))
            model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{trial}_{start_epoch - 1}.pth')))
    model.eval()

    # Evaluate test accuracy
    test_loss, test_acc = test(model, testloader)
    TEST_LOSS_SUM += test_loss
    TEST_ACC_SUM += test_acc

    # compute bias and variance
    bias2, variance = compute_bias_variance(model, testloader, trial)
    variance_unbias = variance * args.trial / (args.trial - 1.0)
    bias2_unbias = TEST_LOSS_SUM / (trial + 1) - variance_unbias
    print('trial: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
        trial, TEST_LOSS_SUM / (trial + 1), TEST_ACC_SUM / (trial + 1), bias2_unbias, variance_unbias))
    log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(
        trial, TEST_LOSS_SUM / (trial + 1), TEST_ACC_SUM / (trial + 1), bias2_unbias, variance_unbias))

print('Program finished')