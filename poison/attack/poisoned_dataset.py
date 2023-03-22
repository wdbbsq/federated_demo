import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os
import torch


class CIFAR10Poison(CIFAR10):
    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            need_idx: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        # 随机选择投毒样本
        self.poi_indices = []
        if need_idx:
            self.poi_indices = generate_poisoned_data(indices, len(self.targets), args.total_workers,
                                                      self.poisoning_rate, args.adversary_list)
        else:
            self.poi_indices = list(random.sample(indices, k=int(len(indices) * self.poisoning_rate)))

        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            # exchange labels
            print(1)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTPoison(MNIST):
    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            need_idx: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        # 随机选择投毒样本
        self.poi_indices = []
        if need_idx:
            self.poi_indices = generate_poisoned_data(indices, len(self.targets), args.total_workers,
                                                      self.poisoning_rate, args.adversary_list)
        else:
            self.poi_indices = list(random.sample(indices, k=int(len(indices) * self.poisoning_rate)))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            # exchange labels
            print(1)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def generate_poisoned_data(indices, data_len, total_workers, poisoning_rate, adversary_list):
    """
    get poisoned data index
    """
    poi_indices = []
    data_pre_client = int(data_len / total_workers)
    for client_id in range(total_workers):
        # 客户端训练集在训练集中的下标范围
        client_indices = indices[client_id * data_pre_client: (client_id + 1) * data_pre_client]
        if client_id in adversary_list:
            poi_indices += list(random.sample(client_indices, k=int(data_pre_client * poisoning_rate)))
    return poi_indices

