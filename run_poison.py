import json
import os
import random
import time
from typing import List

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from utils import get_clients_indices
from utils.serialization import save_as_file
from utils.params import init_parser
from utils.file_utils import prepare_operation

from poison.client import Client
from poison.server import Server
from poison.attack.dataset import build_poisoned_training_sets, build_test_set

LOG_PREFIX = './poison/logs'
LAYER_NAME = '7.weight'

if __name__ == '__main__':

    parser = init_parser('federated poison')

    # poison settings
    parser.add_argument('--poisoning_rate', type=float, default=0.1)
    parser.add_argument('--labels', type=list, default=[1, 9])

    parser.add_argument('--need_scale', type=bool, default=False)
    parser.add_argument('--weight_scale', type=int, default=100, help='恶意更新缩放比例')
    epochs = list(range(40))
    parser.add_argument('--attack_epochs', type=list, default=epochs[19:],
                        help='发起攻击的轮次 默认从15轮训练开始攻击')
    # defense settings
    parser.add_argument('--defense', default='None', help='[None, Flex]')

    args = parser.parse_args()

    args.adversary_list = random.sample(range(args.total_workers), args.adversary_num)
    # 初始化数据集
    train_datasets, args.nb_classes = build_poisoned_training_sets(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_test_set(is_train=False, args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_channels = train_datasets.channels

    clients = []
    clean_clients = []
    evil_clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_datasets, device, i, i in args.adversary_list))
        if i in args.adversary_list:
            evil_clients.append(clients[i])
        else:
            clean_clients.append(clients[i])

    server = Server(args, dataset_val_clean, dataset_val_poisoned, device)

    status = []
    start_time, start_time_str, LOG_PREFIX = prepare_operation(args, LOG_PREFIX)
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

        server.model_aggregate(weight_accumulator)
        test_status = server.evaluate_badnets(device)
        status.append({
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(f"{LOG_PREFIX}/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}.csv",
                  index=False, encoding='utf-8')

    print(f'Fininsh Trainning in {time.time() - start_time}\n ')
