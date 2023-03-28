import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from inference.attack import label_to_onehot, cross_entropy_for_onehot
from skimage.metrics import structural_similarity as ssim
from plot import plot


# 清除cuda缓存
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = datasets.MNIST("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = 25
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0].cpu()))

from inference.models.vision import LeNet, weights_init
from poison.model import get_model

# net = get_model('badnets', device, input_channels=1, output_num=100)
net = LeNet().to(device)

torch.manual_seed(1234)

# net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

# 目标图像的numpy数组
target = gt_data[0][0].cpu().detach().numpy()

data_history = []
label_history = []
ssim_res = []
for iters in range(100):
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
    ssim_res.append(ssim(dummy_data[0][0].cpu().detach().numpy(), target, data_range=target.max() - target.min()))
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        data_history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(9, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data_history[i])
    plt.title(f'迭代次数={(i + 1) * 10}' if i == 0 else f'{(i + 1) * 10}')
    plt.axis('off')

plt.show()

plot(y_data=[ssim_res],
     legends=['推理准确率'],
     colors=['b'],
     linestyles=['-'],
     xlabel='轮次',
     ylabel='$\mathrm{ SSIM }$')