import torch.nn.utils.prune as prune

import torch
from backdoor.model import get_model
from utils import init_model


def clip_by_norm(model, module):
    """
    使用局部非结构化剪枝
    """
    print('ss')


# LAYER_NAME = '7.weight'
# device = torch.device('cuda')
# net = get_model('resnet18', device)
# net = init_model('resnet18')
