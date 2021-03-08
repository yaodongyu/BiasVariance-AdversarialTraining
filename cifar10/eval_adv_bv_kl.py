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
criterion = nn.CrossEntropyLoss().cuda()
nll_loss = nn.NLLLoss(size_average=False)
# variables for bias-variance calculation
NUM_CLASSES = 10
OUTPUTS_LOG_AVG = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
# train/test accuracy/loss
TRAIN_ACC_SUM = 0.0
TEST_ACC_SUM = 0.0
TRAIN_LOSS_SUM = 0.0
TEST_LOSS_SUM = 0.0

def kl_div_cal(P, Q):
    return (P * (P / Q).log()).sum()


def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()

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

        loss = criterion(outputs, targets.long())
        test_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return test_loss / total, 100. * correct / total


def compute_log_output_kl(net, testloader):
    net.eval()
    total = 0
    outputs_log_total = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()

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
        outputs_log_total[total:(total + targets.size(0)), :] = outputs.log()
        total += targets.size(0)
    return outputs_log_total


def compute_normalization_kl(outputs_log_avg):
    outputs_norm = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
    for idx in range(args.test_size):
        for idx_y in range(NUM_CLASSES):
            outputs_norm[idx, idx_y] = torch.exp(outputs_log_avg[idx, idx_y])
    for idx in range(args.test_size):
        y_total = 0.0
        for idx_y in range(NUM_CLASSES):
            y_total += outputs_norm[idx, idx_y]
        outputs_norm[idx, :] /= (y_total * 1.0)
    return outputs_norm


def compute_bias_variance_kl(net, testloader, outputs_avg):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()

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
        bias2 += nll_loss(outputs_avg[total:total + targets.size(0), :].log(), targets.long())
        for idx in range(len(inputs)):
            variance_idx = kl_div_cal(outputs_avg[total + idx], outputs[idx])
            variance += variance_idx
            assert variance_idx > -0.0001
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
    logfilename = os.path.join(args.fname, 'bv_kl_log_best_eps{}.txt'.format(epsilon_file))
else:
    logfilename = os.path.join(args.fname, 'bv_kl_log_epoch{}_eps{}.txt'.format(args.resume, epsilon_file))
init_logfile(logfilename, "trial\ttest loss\ttest acc\tbias2\tvariance")

transform_test = transforms.Compose([transforms.ToTensor(), ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

if not os.path.exists(args.fname):
    os.makedirs(args.fname)

##################################################
# compute log-output average
##################################################
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
        print('loading model: ', os.path.join(args.fname, f'model_{trial}_{start_epoch - 1}.pth'))
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{trial}_{start_epoch - 1}.pth')))
    model.eval()

    OUTPUTS_LOG_AVG += compute_log_output_kl(model, testloader) * (1.0 / args.trial)

##################################################
# normalization
##################################################
OUTPUTS_NORM = compute_normalization_kl(OUTPUTS_LOG_AVG)
variance_total = 0.0

##################################################
# compute bias variance
##################################################
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
        print('loading model: ', os.path.join(args.fname, f'model_{trial}_{start_epoch - 1}.pth'))
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{trial}_{start_epoch - 1}.pth')))
    model.eval()

    test_loss, test_acc = test(model, testloader)
    TEST_LOSS_SUM += test_loss
    TEST_ACC_SUM += test_acc

    # compute bias and variance
    bias2, variance = compute_bias_variance_kl(model, testloader, OUTPUTS_NORM)
    variance_total += variance
    variance_avg = variance_total / (args.trial * 1.0)
    print('trial: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
        trial, TEST_LOSS_SUM / (args.trial * 1.0),
        TEST_ACC_SUM / (args.trial * 1.0), bias2, variance_avg))
    log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(
        trial, TEST_LOSS_SUM / (args.trial * 1.0), TEST_ACC_SUM / (args.trial * 1.0), bias2, variance_avg))

print('Program finished')