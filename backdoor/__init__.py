import argparse
import random
import time
from typing import List

import pandas as pd
import torch

from client import Client
from server import Server
from attack import *
import server


TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Backdoor')

    # base settings
    parser.add_argument('--dataset', default='MNIST',
                        help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--load_local', action='store_true',
                        help='train model or directly load model (default true, if you add this param, then load '
                             'trained local model to evaluate the performance)')
    parser.add_argument('--model_name', default='badnets',
                        help='[badnets, resnet18]')
    parser.add_argument('--loss', default='mse',
                        help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--optimizer', default='sgd',
                        help='Which optimizer to use (sgd or adam, default: sgd)')
    parser.add_argument('--global_epochs', type=int, default=100,
                        help='Number of epochs to train backdoor model, default: 100')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to split dataset, default: 64')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='Batch size to split dataset, default: 64')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='')
    parser.add_argument('--data_path', default='./data/',
                        help='Place to load dataset (default: ./dataset/)')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate of the model, default: 0.001')
    parser.add_argument('--lambda_', type=float, default=0.01,
                        help='')
    parser.add_argument('--momentum', type=float, default=0.0001,
                        help='')

    # federated settings
    parser.add_argument('--total_workers', type=int, default=4)
    parser.add_argument('--k_workers', type=int, default=3,
                        help='clients num selected for each epoch')
    parser.add_argument('--adversary_num', type=int, default=1)
    parser.add_argument('--local_epochs', type=int, default=2)

    # poison settings
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                        help='poisoning portion for local client (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1,
                        help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./backdoor/triggers/trigger_white.png",
                        help='Trigger Path (default: ./backdoor/triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5,
                        help='Trigger Size (int, default: 5)')
    parser.add_argument('--need_scale', type=bool, default=False)
    parser.add_argument('--weight_scale', type=int, default=100,
                        help='恶意更新缩放比例')
    epochs = list(range(40))
    parser.add_argument('--attack_epochs', type=list, default=epochs[29:],
                        help='发起攻击的轮次，默认从15轮训练开始攻击')
    # defense settings
    parser.add_argument('--defense', default='None', help='[None, Flex]')
    # other setting
    parser.add_argument('--need_serialization', type=bool, default=False)

    args = parser.parse_args()

    adversary_list = random.sample(range(args.total_workers), args.adversary_num)
    # 初始化数据集
    train_datasets, args.nb_classes = build_poisoned_training_sets(is_train=True, args=args,
                                                                   adversary_list=adversary_list)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.input_channels = train_datasets.channels
    args.input_channels = 10

    clients = []
    clean_clients = []
    evil_clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_datasets, i, i in adversary_list))
        if i in adversary_list:
            evil_clients.append(clients[i])
        else:
            clean_clients.append(clients[i])

    server = Server(args, dataset_val_clean, dataset_val_poisoned)

    status = []
    start_time = time.time()
    start_time_str = time.strftime(TIME_FORMAT, time.localtime())
    for epoch in range(args.global_epochs):
        # 本轮迭代是否进行攻击
        attack_now = len(args.attack_epochs) != 0 and epoch == args.attack_epochs[0]
        if attack_now:
            args.attack_epochs.pop(0)
            candidates = evil_clients + random.sample(clean_clients, args.k_workers - args.adversary_num)
        else:
            candidates = random.sample(clients, args.k_workers)

        # 客户端上传的模型更新
        weight_accumulator = dict()
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            local_update = c.local_train(server.global_model, epoch, attack_now)
            # 累加客户端更新
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(local_update[name])

        server.model_aggregate(weight_accumulator)
        test_status = server.evaluate_badnets()
        log_status = {
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        }
        status.append(log_status)
        df = pd.DataFrame(status)
        df.to_csv(f"./backdoor/logs/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}_{start_time_str}_trigger{args.trigger_label}.csv",
                  index=False, encoding='utf-8')

    print(f'Fininsh Trainning in {time.time() - start_time}\n ')
