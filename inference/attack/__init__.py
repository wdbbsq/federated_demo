import random

import torch
import torch.nn.functional as F

from torchvision import transforms


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def dlg(net, target_data, target_label, device):
    """
    推理攻击
    :param net:
    :param target_data:
    :param target_label:
    :param device:
    :return:
    """

    criterion = cross_entropy_for_onehot

    dummy_data = torch.randn(target_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(target_label.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    original_dy_dx = list((_.detach().clone() for _ in target_data))

    history = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
