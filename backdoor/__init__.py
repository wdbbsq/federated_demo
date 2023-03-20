import argparse
import json
import random
import time
from typing import List

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from defense.cluster import plot_cluster
from utils import get_clients_indices
from utils.serialization import save_as_file

from client import Client
from server import Server
from attack import *
import server


TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'
LAYER_NAME = '7.weight'

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
    parser.add_argument('--trigger_path', default="./backdoor/triggers/trigger_10.png",
                        help='Trigger Path (default: ./backdoor/triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5,
                        help='Trigger Size (int, default: 5)')
    parser.add_argument('--need_scale', type=bool, default=False)
    parser.add_argument('--weight_scale', type=int, default=100, help='恶意更新缩放比例')
    epochs = list(range(40))
    parser.add_argument('--attack_epochs', type=list, default=epochs[29:],
                        help='发起攻击的轮次 默认从15轮训练开始攻击')
    # defense settings
    parser.add_argument('--defense', default='None', help='[None, Flex]')
    # other setting
    parser.add_argument('--need_serialization', type=bool, default=False)
    parser.add_argument('--platform', default='centos', help='[centos, windows]')

    args = parser.parse_args()

    adversary_list = random.sample(range(args.total_workers), args.adversary_num)
    # 初始化数据集
    train_datasets, args.nb_classes = build_poisoned_training_sets(is_train=True, args=args,
                                                                   adversary_list=adversary_list)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.input_channels = train_datasets.channels
    args.input_channels = 10

    # 不同平台的路径不一样
    LOG_PREFIX = './backdoor/logs' if args.platform == 'centos' else './logs'

    clients = []
    clean_clients = []
    evil_clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_datasets, device, i, i in adversary_list))
        if i in adversary_list:
            evil_clients.append(clients[i])
        else:
            clean_clients.append(clients[i])

    server = Server(args, dataset_val_clean, dataset_val_poisoned, device)

    status = []
    start_time = time.time()
    start_time_str = time.strftime(TIME_FORMAT, time.localtime())
    # 创建文件夹
    LOG_PREFIX = LOG_PREFIX + '/' + start_time_str
    os.makedirs(f'{LOG_PREFIX}')
    # 保存配置参数
    with open(f'{LOG_PREFIX}/params.json', 'wt') as f:
        args.adversary_list = adversary_list
        json.dump(vars(args), f, indent=4)
    f.close()

    for epoch in range(args.global_epochs):
        # 本轮迭代是否进行攻击
        attack_now = len(args.attack_epochs) != 0 and epoch == args.attack_epochs[0]
        if attack_now:
            args.attack_epochs.pop(0)
            candidates = evil_clients + random.sample(clean_clients, args.k_workers - args.adversary_num)
        else:
            candidates = random.sample(clients, args.k_workers)

        client_ids_map = get_clients_indices(candidates)

        # 客户端上传的模型更新
        weight_accumulator = dict()
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        local_updates = []
        for c in candidates:
            local_update = c.local_train(server.global_model, epoch, attack_now)
            local_updates.append({
                'id': c.client_id,
                'local_update': local_update
            })
            # 累加客户端更新
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(local_update[name])

        # 计算余弦相似度
        cos_list = np.zeros([args.k_workers, args.k_workers])
        for i, j in list(combinations(local_updates, 2)):
            cos = cosine_similarity(i['local_update'][LAYER_NAME].reshape(1, -1).cpu().numpy(),
                                    j['local_update'][LAYER_NAME].reshape(1, -1).cpu().numpy())[0][0]
            x, y = client_ids_map.get(i['id']), client_ids_map.get(j['id'])
            cos_list[x][y] = cos
            cos_list[y][x] = cos
        for i in range(args.k_workers):
            cos_list[i][i] = 1

        save_as_file(cos_list, f'{LOG_PREFIX}/{epoch}_cos_numpy')
        # plot_cluster()

        # df = pd.DataFrame(cos_states)
        # df.to_csv(f'./backdoor/logs/cos/epoch{epoch}_{start_time_str}',
        #           index=False, encoding='utf-8')

        server.model_aggregate(weight_accumulator)
        test_status = server.evaluate_badnets(device)
        status.append({
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(f"{LOG_PREFIX}/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}_{start_time_str}_trigger{args.trigger_label}.csv",
                  index=False, encoding='utf-8')

    print(f'Fininsh Trainning in {time.time() - start_time}\n ')
