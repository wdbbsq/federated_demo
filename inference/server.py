import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from utils import init_model


class Server:
    def __init__(self, args, eval_dataset, use_paillier=False):
        self.args = args
        self.eval_loader = DataLoader(eval_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
        if use_paillier:
            net = init_model(args.model_name)
        self.global_model = init_model(args.model_name)

    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * self.args.lambda_
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        # 不启用Batch Normalization和Dropout，
        # 保证测试过程中，Batch Normalization层的均值和方差不变
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            # sum up batch loss
            total_loss += torch.nn.functional.cross_entropy(output,
                                                            target,
                                                            reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = float(correct) / float(dataset_size)
        total_l = total_loss / dataset_size

        return acc, total_l
