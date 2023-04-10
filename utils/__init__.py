import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from .gradient import calculate_model_gradient, calculate_parameter_gradients


def get_dataset(path, name):
    if name == 'MNIST':

        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(path, train=False, transform=transforms.ToTensor())

    elif name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(path, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(path, train=False, transform=transform_test)
    else:
        raise NotImplementedError()

    return train_dataset, eval_dataset


def get_clients_indices(candidates):
    """

    :param candidates:
    :return: Dict[客户端id 下标]
    """
    indices = dict()
    for i, client_id in enumerate(candidates):
        indices[client_id] = i
    return indices
