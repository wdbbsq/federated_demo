import argparse
import random
import time

import torch
import pandas as pd

from client import Client
from server import Server
from utils import get_dataset

TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    # base settings
    parser.add_argument('--dataset', default='CIFAR10',
                        help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--load_local', action='store_true',
                        help='train model or directly load model (default true, if you add this param, then load '
                             'trained local model to evaluate the performance)')
    parser.add_argument('--model_name', default='resnet18',
                        help='[badnets, resnet18]')
    parser.add_argument('--loss', default='mse',
                        help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--optimizer', default='sgd',
                        help='Which optimizer to use (sgd or adam, default: sgd)')
    parser.add_argument('--global_epochs', default=50,
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
    parser.add_argument('--local_epochs', type=int, default=1)

    args = parser.parse_args()

    train_datasets, eval_datasets = get_dataset("./data/", args.dataset)
    server = Server(args, eval_datasets)
    clients = []
    for client_id in range(args.total_workers):
        clients.append(Client(args, train_datasets, client_id))

    status = []
    start_time = time.time()
    start_time_str = time.strftime(TIME_FORMAT, time.localtime())
    for e in range(args.global_epochs):
        candidates = random.sample(clients, args.k_workers)
        # 客户端上传的模型更新
        weight_accumulator = dict()
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            local_update = c.local_train(server.global_model, e)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(local_update[name])

        server.model_aggregate(weight_accumulator)
        acc, loss = server.model_eval()
        log_status = {
            'epoch': e,
            'test_acc': acc,
            'test_loss': loss
        }
        status.append(log_status)
        df = pd.DataFrame(status)
        df.to_csv(f"./baseline/logs/{args.dataset}_{args.model_name}_{start_time_str}.csv",
                  index=False, encoding='utf-8')

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
