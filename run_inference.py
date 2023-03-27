import random
import time

import pandas as pd
import torch

from inference.client import Client
from inference.server import Server
from utils import get_dataset
from utils.file_utils import prepare_operation
from utils.params import init_parser

LOG_PREFIX = './inference/logs'

if __name__ == '__main__':

    parser = init_parser('federated inference')

    # attack settings
    parser.add_argument('--attack', type=bool, default=False)
    # defense settings
    parser.add_argument('--defense', type=bool, default=False)
    args = parser.parse_args()

    args.k_workers = int(args.total_workers * args.global_lr)

    # 初始化数据集
    train_dataset, eval_dataset = get_dataset(args.data_path, args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_channels = 1

    clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_dataset, device, i))

    server = Server(args, eval_dataset, device)

    status = []
    start_time, start_time_str, LOG_PREFIX = prepare_operation(args, LOG_PREFIX)
    for epoch in range(args.global_epochs):
        candidates = random.sample(clients, args.k_workers)

        # 客户端上传的模型更新
        weight_accumulator = dict()
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        local_updates = []
        for c in candidates:
            local_update = c.local_train(server.global_model, epoch)
            # 累加客户端更新
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(local_update[name])

        server.model_aggregate(weight_accumulator)
        test_status = server.eval_model(device, epoch, LOG_PREFIX)
        status.append({
            'epoch': epoch,
            **{f'test_{k}': v for k, v in test_status.items()}
        })
        df = pd.DataFrame(status)
        df.to_csv(
            f"{LOG_PREFIX}/{args.dataset}_{args.model_name}_{args.total_workers}_{args.k_workers}.csv",
            index=False, encoding='utf-8'
        )

    print(f'Finish Training in {time.time() - start_time}\n ')
