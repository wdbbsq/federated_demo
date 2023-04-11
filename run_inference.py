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
    parser.add_argument('--attack', action='store_true')
    # defense settings
    parser.add_argument('--defense', action='store_true')
    args = parser.parse_args()

    args.k_workers = int(args.total_workers * args.global_lr)

    torch.cuda.empty_cache()

    # 初始化数据集
    train_dataset, eval_dataset = get_dataset(args.data_path, args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_channels = 1

    clients = []
    for i in range(args.total_workers):
        clients.append(Client(args, train_dataset, device, i))

    # select leader client
    leader_client: Client = random.sample(clients, 1)[0]
    print(f'leader client: {leader_client.client_id} \n')
    private_key, public_key, enc_model, model_shape = leader_client.init_params(eval_dataset)
    for c in clients:
        if c.client_id != leader_client.client_id:
            c.save_secret(private_key, public_key, model_shape)
    print(f'prepare phase done \n')

    server = Server(args)

    status = []
    start_time, start_time_str, LOG_PREFIX = prepare_operation(args, LOG_PREFIX)
    for epoch in range(args.global_epochs):
        candidates = random.sample(clients, args.k_workers)

        local_updates = []
        for c in candidates:
            c: Client
            local_update = c.boot_training(enc_model, epoch)
            server.accumulate_update(local_update)

        enc_model = server.model_aggregate()
        test_status = leader_client.eval_model(device, epoch, LOG_PREFIX)
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
