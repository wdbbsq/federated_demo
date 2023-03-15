import torch
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


def init_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
