import random
import time

import pandas as pd
import torch

from poison.attack.dataset import build_poisoned_training_sets, build_test_set
from poison.client import Client
from poison.defense import get_clean_updates
from poison.server import Server
from utils import get_clients_indices
from utils.file_utils import prepare_operation
from utils.params import init_parser
from utils.serialization import save_as_file

LOG_PREFIX = './poison/logs'

if __name__ == '__main__':

    parser = init_parser('federated poison')

    # attack settings
    parser.add_argument('--attack', action='store_true')

    parser.add_argument('--poisoning_rate', type=float, default=0.1)
    parser.add_argument('--labels', type=list, default=[1, 9])

    parser.add_argument('--need_scale', action='store_true')
    parser.add_argument('--weight_scale', type=int, default=100, help='恶意更新缩放比例')
    epochs = list(range(20))
    parser.add_argument('--attack_epochs', type=list, default=epochs,
                        help='发起攻击的轮次 默认从15轮训练开始攻击')
    # defense settings
    parser.add_argument('--defense', action='store_true')
    parser.add_argument('--defense_method', default='clique', choices=['clique', 'krum', 'mean'])

    # other
    parser.add_argument('--need_serialization', action='store_true', help='是否保存中间结果')
    args = parser.parse_args()

    args.k_workers = int(args.total_workers * args.global_lr)
    args.adversary_list = random.sample(range(args.total_workers), args.adversary_num) if args.attack else []
    # 初始化数据集
    train_datasets, args.nb_classes = build_poisoned_training_sets(is_train=True, args=args)
    eval_dataset = build_test_set(is_train=False, args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_channels = 1

    clients = []
    clean_clients = []
    evil_clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_datasets, device, i, i in args.adversary_list))
        if i in args.adversary_list:
            evil_clients.append(i)
        else:
            clean_clients.append(i)

    server = Server(args, eval_dataset, device)

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

        server.preparation()
        # 客户端本地训练
        for idx in candidates:
            c: Client = clients[idx]
            local_update = c.boot_training(server.global_model, epoch, attack_now)
            server.sum_update(local_update, c.client_id)
        if args.defense:
            server.apply_defense(client_ids_map, LOG_PREFIX, epoch)

        server.model_aggregate()
        test_status = server.eval_model(server.eval_dataloader, device, epoch, LOG_PREFIX)
        status.append({
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(
            f"{LOG_PREFIX}/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}_Scale{args.need_scale}{args.weight_scale}.csv",
            index=False, encoding='utf-8')

    print(f'Finish Training in {time.time() - start_time}\n ')
