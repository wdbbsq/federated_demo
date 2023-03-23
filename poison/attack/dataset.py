from .poisoned_dataset import MNISTPoison
from torchvision import datasets, transforms
import torch


def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=download)
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data


def build_poisoned_training_sets(is_train, args):
    transform, detransform = build_transform(args.dataset, is_train=True)

    if args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True,
                               transform=transform, need_idx=True)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes


def build_test_set(is_train, args):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return testset_clean, testset_poisoned


def build_transform(dataset, is_train=False):
    if dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(),
                                       (1.0 / std).tolist())  # you can use detransform to recover the image

    return transform, detransform
