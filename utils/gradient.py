import numpy
from scipy.spatial import distance


def calc_dist(model_dict_a, model_dict_b, layer_name):
    return distance.cdist(get_vector(model_dict_a, layer_name),
                          get_vector(model_dict_b, layer_name), "euclidean")[0][0]


def get_vector(model_dict, layer_name):
    return model_dict[layer_name].reshape(1, -1).cpu().numpy()


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
