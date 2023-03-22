import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os
import torch


class TriggerHandler(object):
    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


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
            adversary_list: list = [],
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        # 随机选择投毒样本
        self.poi_indices = []
        if need_idx:
            self.poi_indices = generate_poisoned_data(indices, len(self.targets), args.total_workers,
                                                      self.poisoning_rate, adversary_list)
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
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

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
            adversary_list: list = [],
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        # 随机选择投毒样本
        self.poi_indices = []
        if need_idx:
            self.poi_indices = generate_poisoned_data(indices, len(self.targets), args.total_workers,
                                                      self.poisoning_rate, adversary_list)
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
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

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


"""
methods from `How to backdoor federated learning`
"""


def poison_train_dataset(conf, train_dataset):
    """
    生成中毒数据集
    :param train_dataset:
    :param conf:
    :return:
    """
    #
    # return [(self.train_dataset[self.params['poison_image_id']][0],
    # torch.IntTensor(self.params['poison_label_swap']))]
    cifar_classes = {}
    for ind, x in enumerate(train_dataset):
        _, label = x
        if ind in conf.params['poison_images'] or ind in conf.params['poison_images_test']:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    indices = list()
    # create array that starts with poisoned images

    # create candidates:
    # range_no_id = cifar_classes[1]
    # range_no_id.extend(cifar_classes[1])

    # 剔除数据集中的后门样本
    range_no_id = list(range(50000))
    for image in conf.params['poison_images'] + conf.params['poison_images_test']:
        if image in range_no_id:
            range_no_id.remove(image)

    # add random images to other parts of the batch
    for batches in range(0, conf.params['size_of_secret_dataset']):
        range_iter = random.sample(range_no_id, conf.params['batch_size'])
        # range_iter[0] = self.params['poison_images'][0]
        indices.extend(range_iter)
        # range_iter = random.sample(range_no_id,
        #            self.params['batch_size']
        #                -len(self.params['poison_images'])*self.params['poisoning_per_batch'])
        # for i in range(0, self.params['poisoning_per_batch']):
        #     indices.extend(self.params['poison_images'])
        # indices.extend(range_iter)
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=conf.params['batch_size'],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
                                       )


def poison_test_dataset(conf, train_dataset):
    #
    # return [(self.train_dataset[self.params['poison_image_id']][0],
    # torch.IntTensor(self.params['poison_label_swap']))]
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=conf.params['batch_size'],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                           range(1000))
                                       )
