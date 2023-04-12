from typing import Dict

import numpy
import torch
from scipy.spatial import distance


def scale_update(weight_scale, local_update: Dict[str, torch.Tensor]):
    """
    按比例缩放参数
    """
    if weight_scale == 1:
        return
    for name, value in local_update.items():
        if type(weight_scale) != int and value.dtype == torch.int64:
            continue
        value.mul_(weight_scale)


def calc_dist(model_dict_a, model_dict_b, layer_name):
    """
    计算模型某一层的欧式距离
    """
    return calc_vector_dist(get_vector(model_dict_a, layer_name),
                            get_vector(model_dict_b, layer_name))


def calc_vector_dist(vec_1, vec_2):
    return distance.cdist(vec_1, vec_2, "euclidean")[0][0]


def get_vector(model_dict, layer_name):
    """
    将输入Tensor转换为1维
    """
    return model_dict[layer_name].reshape(1, -1).cpu().numpy()


def get_grads(model, loss):
    """
    计算模型梯度
    """
    grads = list(torch.autograd.grad(loss.mean(),
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True))
    return grads


def calculate_model_gradient(model_1, model_2):
    """
    Calculates the gradient (parameter difference) between two Torch models.

    :param model_1: torch.nn
    :param model_2: torch.nn
    """
    model_1_parameters = list(dict(model_1.state_dict()))
    model_2_parameters = list(dict(model_2.state_dict()))

    return calculate_parameter_gradients(model_1_parameters, model_2_parameters)


def calculate_parameter_gradients(params_1, params_2):
    """
    Calculates the gradient (parameter difference) between two sets of Torch parameters.

    :param model_1: dict
    :param model_2: dict
    """

    return numpy.array([x for x in numpy.subtract(params_1, params_2)])
