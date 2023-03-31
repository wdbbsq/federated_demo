import argparse
import random

import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

from attack.poisoned_dataset import CIFAR10Poison
from utils.params import init_parser


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataiter=iter(trainloader)
# images,labels=dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(''.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    # base settings
    parser = init_parser('federated backdoor')

    # poison settings
    parser.add_argument('--attack', type=bool, default=True, help='是否进行攻击')
    parser.add_argument('--attack_method', default='dba', help='攻击类型：[central, dba]')
    parser.add_argument('--poisoning_rate', type=float, default=1.0,
                        help='poisoning portion for local client (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1,
                        help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./triggers/trigger_10",
                        help='触发器路径，不含文件扩展名')
    parser.add_argument('--trigger_size', type=int, default=6,
                        help='Trigger Size (int, default: 5)')

    args = parser.parse_args()
    args.total_workers = 16
    args.adversary_num = 4
    args.k_workers = int(args.total_workers * args.global_lr)
    args.adversary_list = [0, 1, 2, 3]
    # args.adversary_list = random.sample(range(args.total_workers), args.adversary_num) if args.attack else []

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # testset = datasets.CIFAR10(args.data_path, download=True, transform=transform)
    train_set = CIFAR10Poison(args, args.data_path, train=True, download=True, transform=transform)
    test_set = CIFAR10Poison(args, args.data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print('ok')

    # 训练集
    triggers = [2840, 4697, 7376, 11974]
    train_img = torch.stack([train_set.__getitem__(i)[0] for i in triggers])
    imshow(torchvision.utils.make_grid(train_img))
    # 测试集
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))


