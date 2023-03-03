import argparse
import random

import yaml
# from datasets.data_loader_utils import *
from utils.distributaion import *
from backdoor.backdoor.backdoor_client import *
from backdoor.backdoor.attack import *
from baseline.server import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Backdoor')
    
    # base settings
    parser.add_argument('--dataset', default='MNIST', 
                        help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--nb_classes', default=10, type=int, 
                        help='number of the classification types')
    parser.add_argument('--load_local', action='store_true',
                        help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
    parser.add_argument('--loss', default='mse', 
                        help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--optimizer', default='sgd', 
                        help='Which optimizer to use (sgd or adam, default: sgd)')
    parser.add_argument('--global_epochs', default=100, 
                        help='Number of epochs to train backdoor model, default: 100')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size to split dataset, default: 64')
    parser.add_argument('--test_batch_size', type=int, default=64, 
                        help='Batch size to split dataset, default: 64')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Batch size to split dataset, default: 64')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate of the model, default: 0.001')
    parser.add_argument('--download', action='store_true',
                        help='Do you want to download data ( default false, if you add this param, then download)')
    parser.add_argument('--data_path', default='./data/', 
                        help='Place to load dataset (default: ./dataset/)')
    
    # fedeerated settings
    parser.add_argument('--total_num', type='int', default=4)
    parser.add_argument('--k_workers', type='int', default=3, 
                        help='clients num selected for each epoch')
    parser.add_argument('--adversary_num', type='int', default=1)
    parser.add_argument('--local_epoch', type='int', default=2)

    # poison settings 
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                        help='poisoning portion for local client (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1,
                        help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./triggers/trigger_white.png",
                        help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, 
                        help='Trigger Size (int, default: 5)')
    args = parser.parse_args()


    adversary_list = random.sample(range(args.total_num), args.adversary_num)

    train_datasets, args.nb_classes = build_poisoned_training_sets(is_train=True, args=args, adversary_list=adversary_list)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    clients = []
    for i in range(args.total_num):
        clients.append(Client(args, train_datasets, i, i in adversary_list))

    server = Server(args, dataset_val_clean)
    print("\n\n")

    for e in range(args.global_epochs):
        candidates = random.sample(clients, args.k_workers)
        # 客户端上传的模型更新
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            local_update = c.local_train(server.global_model)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(local_update[name])

        server.model_aggregate(weight_accumulator)
        acc, loss = server.model_eval()
        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

