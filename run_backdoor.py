import random
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from backdoor.attack.dataset import build_training_sets, build_test_set
from backdoor.client import Client
from backdoor.server import Server
from utils import get_clients_indices
from utils.file_utils import prepare_operation
from utils.params import init_parser

LOG_PREFIX = './backdoor/logs'
LAYER_NAME = 'fc.weight'

if __name__ == '__main__':

    parser = init_parser('federated backdoor')

    # poison settings
    parser.add_argument('--attack', action='store_true', help='是否进行攻击')
    parser.add_argument('--attack_method', default='central', help='攻击类型：[central, dba]')
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                        help='poisoning portion for local client (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1,
                        help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./backdoor/triggers/trigger_white",
                        help='触发器路径，不含文件扩展名')
    parser.add_argument('--trigger_size', type=int, default=5,
                        help='Trigger Size (int, default: 5)')
    parser.add_argument('--need_scale', action='store_true', help='是否缩放参数')
    parser.add_argument('--weight_scale', type=int, default=100, help='恶意更新缩放比例')
    epochs = list(range(40))
    parser.add_argument('--attack_epochs', type=list, default=epochs[9:],
                        help='发起攻击的轮次 默认从15轮训练开始攻击')
    # defense settings
    parser.add_argument('--defense', action='store_true', help='是否防御')
    parser.add_argument('--cluster', action='store_true', help='是否聚类')
    parser.add_argument('--clip', action='store_true', help='是否剪枝')
    # other setting
    parser.add_argument('--need_serialization', action='store_true', help='是否保存训练中间结果')

    args = parser.parse_args()

    args.k_workers = int(args.total_workers * args.global_lr)
    args.adversary_list = random.sample(range(args.total_workers), args.adversary_num) if args.attack else []

    torch.cuda.empty_cache()

    # 初始化数据集
    clean_train_set, poisoned_train_set = build_training_sets(is_train=True, args=args)
    clean_eval_dataset, poisoned_eval_dataset = build_test_set(is_train=False, args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_channels = poisoned_train_set.channels

    clients = []
    clean_clients = []
    evil_clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, poisoned_train_set, clean_train_set, device, i, i in args.adversary_list))
        if i in args.adversary_list:
            evil_clients.append(i)
        else:
            clean_clients.append(i)

    server = Server(args, clean_eval_dataset, poisoned_eval_dataset, device)

    status = []
    start_time, start_time_str, LOG_PREFIX = prepare_operation(args, LOG_PREFIX)
    for epoch in range(args.global_epochs):
        server.preparation()
        # 本轮迭代是否进行攻击
        attack_now = args.attack and len(args.attack_epochs) != 0 and epoch == args.attack_epochs[0]
        if attack_now:
            args.attack_epochs.pop(0)
            candidates = evil_clients + random.sample(clean_clients, args.k_workers - args.adversary_num)
        else:
            candidates = random.sample(clean_clients, args.k_workers)

        client_ids_map = get_clients_indices(candidates)

        for idx in candidates:
            c = clients[idx]
            local_update = c.boot_training(server.global_model, epoch, attack_now)
            server.sum_update(local_update, idx)

        # 进行防御
        if args.defense:
            server.apply_defense(LAYER_NAME, args.k_workers, client_ids_map)

        server.model_aggregate()
        test_status = server.evaluate_backdoor(device, epoch, LOG_PREFIX) if args.attack else \
            server.eval_model(server.eval_dataloader, device, epoch, LOG_PREFIX)

        status.append({
            'epoch': epoch,
            **{f'{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(f"{LOG_PREFIX}/{args.attack_method}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}_trigger{args.trigger_label}.csv",
                  index=False, encoding='utf-8')

    print(f'Finish Training in {time.time() - start_time}\n ')
