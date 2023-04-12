import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from utils.serialization import save_as_file
from utils.eval import model_evaluation


class BaseServer:
    def __init__(self, args, eval_dataset, device):
        self.args = args
        self.global_model = get_model(args.model_name,
                                      device,
                                      input_channels=args.input_channels,
                                      output_num=args.nb_classes)
        self.eval_dataloader = DataLoader(eval_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)
        # 保存客户端上传的模型更新
        self.weight_accumulator = dict()

    def preparation(self):
        """
        每轮训练前的准备工作
        """
        # 重新初始化accumulator
        self.weight_accumulator = dict()
        for name, params in self.global_model.state_dict().items():
            self.weight_accumulator[name] = torch.zeros_like(params)

    def sum_update(self, client_update):
        """
        累加客户端更新
        """
        for name, params in self.global_model.state_dict().items():
            self.weight_accumulator[name].add_(client_update[name])

    def model_aggregate(self):
        """
        聚合客户端更新
        """
        # for name, sum_update in weight_accumulator.items():
        #     scale = self.args.k_workers / self.args.total_workers
        #     average_update = scale * sum_update
        #     model_weight = self.global_model.state_dict()[name]
        #     if model_weight.type() == average_update.type():
        #         model_weight.add_(average_update)
        #     else:
        #         model_weight.add_(average_update.to(torch.int64))

        for name, data in self.global_model.state_dict().items():
            update_per_layer = self.weight_accumulator[name] * self.args.lambda_
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def eval_model(self, data_loader, device, epoch, file_path):
        """
        测试模型性能并返回评估值
        """
        return model_evaluation(self.global_model, data_loader, device, file_path,
                                epoch == self.args.global_epochs - 1)
