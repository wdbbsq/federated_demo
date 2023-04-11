from .poisoned_dataset import CIFAR10Poison, MNISTPoison
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


def build_training_sets(is_train, args):
    transform, de_transform = build_transform(args.dataset, is_train=True)

    if args.dataset == 'CIFAR10':
        clean_training_set = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        poisoned_training_set = CIFAR10Poison(args, args.data_path, train=is_train, download=True,
                                     transform=transform)
    elif args.dataset == 'MNIST':
        clean_training_set = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        poisoned_training_set = MNISTPoison(args, args.data_path, train=is_train, download=True,
                                   transform=transform)
    else:
        raise NotImplementedError()

    print(poisoned_training_set)

    return clean_training_set, poisoned_training_set


def build_test_set(is_train, args):
    transform, detransform = build_transform(args.dataset)

    if args.dataset == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    else:
        raise NotImplementedError()

    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned


def build_transform(dataset, is_train=False):
    if dataset == "CIFAR10":
        if is_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            return transform_train, None
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            return transform_test, None

        # transforms.Resize(size):将图片的短边缩放成size的比例，然后长边也跟着缩放，使得缩放后的图片相对于原图的长宽比不变
        # transforms.CenterCrop(size):从图像的中心位置裁剪指定大小的图像
        # ToTensor():将图像由PIL转换为Tensor
        # transform.Normalize():把0-1变换到(-1,1)
        # image = (image - mean) / std
        # 其中mean和std分别通过(0.5, 0.5, 0.5)和(0.2, 0.2, 0.2)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1
        # trans = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5),(0.2, 0.2, 0.2))
        # ])
        # return trans, None
    elif dataset == "MNIST":
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
