import argparse


def init_parser(description):
    parser = argparse.ArgumentParser(description=description)

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

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda_', type=float, default=0.01, help='')
    parser.add_argument('--momentum', type=float, default=0.0001, help='')

    # federated settings
    parser.add_argument('--total_workers', type=int, default=4)
    parser.add_argument('--k_workers', type=int, default=3,
                        help='clients num selected for each epoch')
    parser.add_argument('--adversary_num', type=int, default=1)
    parser.add_argument('--local_epochs', type=int, default=2)

    return parser
