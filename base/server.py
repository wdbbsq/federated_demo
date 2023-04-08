from abc import abstractmethod
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


class BaseServer:
    def __init__(self, args, global_model, dataset_val_clean, dataset_val_poisoned, device):
        self.args = args
        self.global_model = global_model
        self.loader_val_clean = DataLoader(dataset_val_clean,
                                           batch_size=args.batch_size,
                                           shuffle=True)
        self.loader_val_poisoned = DataLoader(dataset_val_poisoned,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    def model_aggregate(self, weight_accumulator):
        # for name, sum_update in weight_accumulator.items():
        #     scale = self.args.k_workers / self.args.total_workers
        #     average_update = scale * sum_update
        #     model_weight = self.global_model.state_dict()[name]
        #     if model_weight.type() == average_update.type():
        #         model_weight.add_(average_update)
        #     else:
        #         model_weight.add_(average_update.to(torch.int64))

        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * self.args.lambda_
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def eval(self, data_loader, model, device, print_perform=False):
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()  # switch to eval status
        y_true = []
        y_predict = []
        loss_sum = []
        with torch.no_grad():
            for (batch_x, batch_y) in tqdm(data_loader):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                batch_y_predict = model(batch_x)

                loss = criterion(batch_y_predict, batch_y)
                batch_y_predict = torch.argmax(batch_y_predict, dim=1)
                y_true.append(batch_y)
                y_predict.append(batch_y_predict)
                loss_sum.append(loss.item())

        y_true = torch.cat(y_true, 0)
        y_predict = torch.cat(y_predict, 0)
        loss = sum(loss_sum) / len(loss_sum)

        if print_perform:
            print(classification_report(y_true.cpu(), y_predict.cpu(),
                                        target_names=data_loader.dataset.classes))

        return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
        }

    def evaluate_badnets(self, device):
        mta = self.eval(self.loader_val_clean, self.global_model, device, print_perform=True)
        bta = self.eval(self.loader_val_poisoned, self.global_model, device, print_perform=False)
        # mta = model_eval(self.global_model, self.loader_val_clean, device)
        # bta = model_eval(self.global_model, self.loader_val_poisoned, device)
        return {
            'clean_acc': mta['acc'], 'clean_loss': mta['loss'],
            'bta': bta['acc'], 'asr_loss': bta['loss'],
        }

    @abstractmethod
    def apply_defense(self):
        """
        进行防御
        """
        pass
