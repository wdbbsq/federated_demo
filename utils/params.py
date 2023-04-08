import argparse


def init_parser(description):
    parser = argparse.ArgumentParser(description=description)

    # base settings
    parser.add_argument('--dataset', default='MNIST',
                        help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--model_name', default='badnets', help='[badnets, resnet18]')

    # 保存模型
    parser.add_argument('--load_local', action='store_true', help='使用保存的全局模型开始训练')
    parser.add_argument('--model_path', default='', help='保存的模型路径')
    parser.add_argument('--start_epoch', type=int, default=0, help='继续训练的全局轮次')

    parser.add_argument('--loss', default='mse')
    parser.add_argument('--optimizer', default='sgd', help='[sgd, adam]')
    parser.add_argument('--global_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2, help='')
    parser.add_argument('--data_path', default='~/.torch',
                        help='Place to load dataset (default: ~/.torch)')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda_', type=float, default=0.01, help='')
    parser.add_argument('--momentum', type=float, default=0.0001, help='')

    # federated settings
    parser.add_argument('--total_workers', type=int, default=4)
    parser.add_argument('--global_lr', type=float, default=0.75)

    parser.add_argument('--adversary_num', type=int, default=1)
    parser.add_argument('--local_epochs', type=int, default=2)

    # 数据分布
    parser.add_argument('--no_iid', action='store_true')
    '''
    使用Dirichlet分布模拟no-iid，
    https://zhuanlan.zhihu.com/p/468992765
    '''
    parser.add_argument('--alpha', type=float, default=0.1)

    return parser
