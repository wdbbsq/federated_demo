import torch.nn.utils.prune as prune

import torch
from models import get_model
import numpy as np
from utils.gradient import calc_dist
from torch.nn.utils.prune import BasePruningMethod


class MyPruning(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        return mask


def clip_by_norm(model):
    """
    使用局部非结构化剪枝
    """
    # Prune 30% of the weights with the lowest L1-norm
    prune.l1_unstructured(model.fc, 'weight', amount=0.3)
    prune.custom_from_mask()


def compute_norms(global_dict, local_updates, layer_name):
    """
    计算与全局模型的L2范数
    """
    norm_list = []
    for update in local_updates:
        norm_list.append(calc_dist(global_dict, update['local_update'], layer_name))
    median_l2_norm = np.median(norm_list)
    return norm_list, median_l2_norm


# LAYER_NAME = '7.weight'
# device = torch.device('cuda')
# net = get_model('badnets', device, input_channels=1, output_num=10)
# net = init_model('resnet18')
# print(net)
