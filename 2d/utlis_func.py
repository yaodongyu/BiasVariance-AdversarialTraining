import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn
import matplotlib.patches as patches
from scipy.spatial import HalfspaceIntersection
import numpy as np

def adv_loss(model,
             x_natural,
             y,
             optimizer,
             step_size=0.003,
             epsilon=0.031,
             perturb_steps=10):
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.rand(x_natural.shape).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_adv = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_adv, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
    model.train()

    x_adv = Variable(x_adv, requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss_adv = F.cross_entropy(logits, y)
    err = (logits.max(1)[1].data != y).float().mean()
    return loss_adv, err


def standard_train(X, y):
    net = nn.Sequential(
        nn.Linear(2,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,2)
    )

    opt = optim.Adam(net.parameters(), lr=1e-3)
    for i in range(1000):
        out = net(Variable(X))
        l = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1].data != y).float().mean()
        if i % 100 == 0:
            print('CE Loss: %.5f,  Error: %.5f'%(l.data.item(), err.item()))
        opt.zero_grad()
        (l).backward()
        opt.step()
    return net.eval()


def robust_train(X, y, epsilon, seed='no'):
    if seed != 'no':
        torch.manual_seed(seed)
    robust_net = nn.Sequential(
        nn.Linear(2,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,2)
    )
    data = []
    opt = optim.Adam(robust_net.parameters(), lr=1e-3)
    robust_net.train()
    for i in range(2000):
        robust_ce, robust_err = adv_loss(model=robust_net,
                                         x_natural=Variable(X),
                                         y=Variable(y),
                                         optimizer=opt,
                                         step_size=epsilon * 0.4,
                                         epsilon=epsilon,
                                         perturb_steps=10)
        out = robust_net(X)
        l2 = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1].data != y).float().mean()
        data.append([l2.data.item(), robust_ce.data.item(), err, robust_err])
        opt.zero_grad()
        (robust_ce).backward()
        opt.step()

    print('Robust CE Loss: %.5f,  Robust Error: %.5f'%(robust_ce.data.item(), robust_err.item()))
    return robust_net


def generate_sample(m, margin, seed='no'):
    if seed != 'no':
        np.seed(seed)

    x_0 = []
    i = 0

    while(len(x_0) < m):
        i+=1
        if i > 50000:
            break
        p = np.random.uniform(size=(2))*2 - np.array([1, 1])
        if np.abs(p[0]-p[1]) > margin:
            x_0.append(p)

    X_0 = torch.Tensor(np.array(x_0))
    torch.manual_seed(1)
    y_0 = (X_0[:, 0] > X_0[:, 1]).type(torch.uint8).long()
    return (X_0, y_0)


def visualize(net, X, y, margin, r):
    XX, YY = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
    y0 = net(X0)
    ZZ = (y0[:,0] - y0[:,1]).resize(100,100).data.numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-1000,1000,3))
    ax.scatter(X.numpy()[:,0], X.numpy()[:,1], c=y.numpy(), cmap="coolwarm", s=70)
    ax.plot([-1.0, 1.0], [-1.0 + margin, 1.0 + margin], '--', color='black')
    ax.plot([-1.0, 1.0], [-1.0 - margin, 1.0 - margin], '--', color='black')
    ax.axis("equal")
    ax.axis([-1,1,-1,1])

    for a in X.numpy():
        ax.add_patch(patches.Rectangle((a[0]-r/2, a[1]-r/2), r, r, fill=False))


def visualize_data(X, y, margin):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X.numpy()[:,0], X.numpy()[:,1], c=y.numpy(), cmap="coolwarm", s=2)
    ax.plot([-1.0, 1.0], [-1.0 + margin, 1.0 + margin], '--', color='black')
    ax.plot([-1.0, 1.0], [-1.0 - margin, 1.0 - margin], '--', color='black')
    ax.axis("equal")
    ax.axis([-1,1,-1,1])


def BV_compute(net_0, net_1, X_test, y_test):
    NUM_CLASSES = 2
    mse = nn.MSELoss(reduction='mean')
    targets_onehot = torch.FloatTensor(y_test.size(0), NUM_CLASSES)
    targets_onehot.zero_()
    targets_onehot.scatter_(1, y_test.view(-1, 1).long(), 1)
    targets_onehot = targets_onehot[:,0]
    y_hat_0 = F.softmax(net_0(X_test), dim=1)[:,0]
    y_hat_1 = F.softmax(net_1(X_test), dim=1)[:,0]
    y_hat_avg = (y_hat_0 + y_hat_1) * 0.5
    bias2 = mse(y_hat_avg, targets_onehot) * 2.0
    variance =  0.5 * mse(y_hat_1, y_hat_0)
    risk = 2.0 * (0.5 * mse(y_hat_0, targets_onehot) + 0.5 * mse(y_hat_1, targets_onehot))
    assert abs(bias2.item() + variance.item() - risk.item()) < 1e-4, "Error computing Bias/Variance."
    return bias2.item(), variance.item(), risk.item()

