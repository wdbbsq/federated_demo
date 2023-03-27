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
from poison.defense import get_clean_updates

LOG_PREFIX = './poison/logs'
LAYER_NAME = '7.weight'

if __name__ == '__main__':

    parser = init_parser('federated poison')

    # attack settings
    parser.add_argument('--attack', type=bool, default=False)

    parser.add_argument('--poisoning_rate', type=float, default=0.1)
    parser.add_argument('--labels', type=list, default=[1, 9])

    parser.add_argument('--need_scale', type=bool, default=False)
    parser.add_argument('--weight_scale', type=int, default=100, help='恶意更新缩放比例')
    epochs = list(range(20))
    parser.add_argument('--attack_epochs', type=list, default=epochs,
                        help='发起攻击的轮次 默认从15轮训练开始攻击')
    # defense settings
    parser.add_argument('--defense', type=bool, default='False')
    parser.add_argument('--defense_method', default='clique', help='[clique, krum, mean]')

    # other
    parser.add_argument('--need_serialization', type=bool, default=False)
    args = parser.parse_args()

    args.k_workers = int(args.total_workers * args.global_lr)
    args.adversary_list = random.sample(range(args.total_workers), args.adversary_num) if args.attack else []
    # 初始化数据集
    train_datasets, args.nb_classes = build_poisoned_training_sets(is_train=True, args=args)
    dataset_val_clean = build_test_set(is_train=False, args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_channels = 1

    clients = []
    clean_clients = []
    evil_clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_datasets, device, i, i in args.adversary_list))
        if i in args.adversary_list:
            evil_clients.append(clients[i])
        else:
            clean_clients.append(clients[i])

    server = Server(args, dataset_val_clean, device)

    status = []
    start_time, start_time_str, LOG_PREFIX = prepare_operation(args, LOG_PREFIX)
    for epoch in range(args.global_epochs):
        # 本轮迭代是否进行攻击
        attack_now = args.attack and len(args.attack_epochs) != 0 and epoch == args.attack_epochs[0]
        if attack_now:
            args.attack_epochs.pop(0)
            candidates = evil_clients + random.sample(clean_clients, args.k_workers - args.adversary_num)
        else:
            # 本轮不攻击，则保证没有恶意客户端
            candidates = random.sample(clean_clients, args.k_workers)

        client_ids_map = get_clients_indices(candidates)

        # 客户端上传的模型更新
        weight_accumulator = dict()
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        local_updates = []
        for c in candidates:
            local_update = c.local_train(server.global_model, epoch, attack_now)
            if args.defense:
                local_updates.append({
                    'id': c.client_id,
                    'local_update': local_update
                })
            else:
                # 累加客户端更新
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(local_update[name])
        if args.defense:
            clean_nodes = get_clean_updates(local_updates, args.defense_method)
            for idx in clean_nodes:
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(local_updates[client_ids_map.get(idx)]['local_update'][name])
            if args.need_serialization:
                save_as_file(local_updates, f'{LOG_PREFIX}/{epoch}_dist')

        server.model_aggregate(weight_accumulator)
        test_status = server.eval_model(device, epoch, LOG_PREFIX)
        status.append({
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(
            f"{LOG_PREFIX}/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}.csv",
            index=False, encoding='utf-8')

    print(f'Fininsh Trainning in {time.time() - start_time}\n ')
