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

from backdoor.client import Client
from backdoor.server import Server
from backdoor.attack.dataset import build_poisoned_training_sets, build_testset
from backdoor.defense.clip import clip_by_norm


TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'
LAYER_NAME = '7.weight'

if __name__ == '__main__':

    parser = init_parser('federated backdoor')

    # poison settings
    parser.add_argument('--attack_type', default='central')
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
    parser.add_argument('--attack_epochs', type=list, default=epochs[19:],
                        help='发起攻击的轮次 默认从15轮训练开始攻击')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.input_channels = train_datasets.channels
    args.input_channels = 10

    LOG_PREFIX = './backdoor/logs'

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

        if attack_now and c.client_id in adversary_list:
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

            save_as_file({
                'cos_list': cos_list,
                'client_ids_map': client_ids_map
            }, f'{LOG_PREFIX}/{epoch}_cos_numpy')

        if args.defense is not None:
            clip_by_norm(server.global_model, LAYER_NAME)

        server.model_aggregate(weight_accumulator)
        test_status = server.evaluate_badnets(device)
        status.append({
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(f"{LOG_PREFIX}/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}_trigger{args.trigger_label}.csv",
                  index=False, encoding='utf-8')

    print(f'Fininsh Trainning in {time.time() - start_time}\n ')
