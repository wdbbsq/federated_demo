import torch.nn.utils.prune as prune

import torch
from backdoor.model import get_model
from utils import init_model
import numpy as np
from utils.gradient import calc_dist


def clip_by_norm(model):
    """
    使用局部非结构化剪枝
    """
    # Prune 30% of the weights with the lowest L1-norm
    prune.l1_unstructured(model.fc, 'weight', amount=0.3)


def clip_clients(global_dict, local_updates, layer_name):
    """
    遍历并裁剪客户端更新
    """
    dist = []
    for update in local_updates:
        dist.append(calc_dist(global_dict, update['local_update'], layer_name))
    median_dist = np.median(dist)


# LAYER_NAME = '7.weight'
device = torch.device('cuda')
net = get_model('badnets', device, input_channels=1, output_num=10)
# net = init_model('resnet18')
print(net)
