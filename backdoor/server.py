import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import model


class Server:
    def __init__(self, args, dataset_val_clean, dataset_val_poisoned):
        self.args = args
        self.global_model = model.get_model(args.model_name,
                                            args.device,
                                            input_channels=args.input_channels,
                                            output_num=args.nb_classes)
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
        for (batch_x, batch_y) in tqdm(data_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            if self.args.dataset == 'CIFAR10':
                # batch_y_predict, out_1000 = model(batch_x)
                batch_y_predict = model(batch_x)
            elif self.args.dataset == 'MNIST':
                batch_y_predict = model(batch_x)
            else:        
                raise NotImplementedError(f'Unkown dataset {self.args.dataset}')
            
            loss = criterion(batch_y_predict, batch_y)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_true.append(batch_y)
            y_predict.append(batch_y_predict)
            loss_sum.append(loss.item())

        y_true = torch.cat(y_true, 0)
        y_predict = torch.cat(y_predict, 0)
        loss = sum(loss_sum) / len(loss_sum)

        if print_perform:
            print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

        return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
        }

    def evaluate_badnets(self):
        mta = self.eval(self.loader_val_clean, self.global_model, self.args.device, print_perform=True)
        bta = self.eval(self.loader_val_poisoned, self.global_model, self.args.device, print_perform=False)
        # mta = model_eval(self.global_model, self.loader_val_clean, self.args.device)
        # bta = model_eval(self.global_model, self.loader_val_poisoned, self.args.device)
        return {
            'clean_acc': mta['acc'], 'clean_loss': mta['loss'],
            'bta': bta['acc'], 'asr_loss': bta['loss'],
        }

def model_eval(global_model, data_loader, device):
    # 不启用Batch Normalization和Dropout，
    # 保证测试过程中，Batch Normalization层的均值和方差不变
    global_model.eval()

    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = global_model(data)
        # sum up batch loss
        total_loss += torch.nn.functional.cross_entropy(output,
                                                        target,
                                                        reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    return {
        'acc': (float(correct) / float(dataset_size)),
        'loss': total_loss / dataset_size
    }