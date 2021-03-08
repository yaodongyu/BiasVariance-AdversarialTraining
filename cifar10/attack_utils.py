import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision

upper_limit, lower_limit = 1, 0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, norm):
    model.eval()
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(X + delta)
        index = slice(None, None, None)
        if not isinstance(index, slice) and len(index) == 0:
            break

        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = delta[index, :, :, :]
        g = grad[index, :, :, :]
        x = X[index, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[index, :, :, :] = d
        delta.grad.zero_()

    model.train()
    return delta.detach()


def attack_pgd_eval(model, X, y, epsilon, alpha, attack_iters, norm):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(X + delta)
        index = slice(None, None, None)
        if not isinstance(index, slice) and len(index) == 0:
            break

        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = delta[index, :, :, :]
        g = grad[index, :, :, :]
        x = X[index, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[index, :, :, :] = d
        delta.grad.zero_()

    return delta.detach()


def attack_pgd_bv_eval(model, X, y, epsilon, alpha, attack_iters, norm):
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(X + delta)
        index = slice(None, None, None)
        if not isinstance(index, slice) and len(index) == 0:
            break

        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = delta[index, :, :, :]
        g = grad[index, :, :, :]
        x = X[index, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[index, :, :, :] = d
        delta.grad.zero_()

    return delta.detach()
