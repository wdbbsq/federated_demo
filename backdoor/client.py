from tqdm import tqdm
from typing import Dict
import model

import torch
from torch.utils.data import DataLoader


class Client:

    def __init__(self, args, train_dataset, client_id=-1, is_adversary=False):
        self.args = args
        self.device = args.device
        self.local_epochs = args.local_epochs
        self.local_model = model.get_model(args.model_name,
                                           args.device,
                                           input_channels=args.input_channels,
                                           output_num=args.nb_classes)
        self.client_id = client_id
        self.is_adversary = is_adversary

        all_range = list(range(len(train_dataset)))
        data_len = int(len(train_dataset) / args.total_workers)
        train_indices = all_range[client_id * data_len: (client_id + 1) * data_len]
        # todo 每个客户端自己的验证集
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
                                       )

    def local_train(self, global_model, global_epoch, attack_now=False):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum)
        self.local_model.train()

        for epoch in range(self.local_epochs):
            for batch_id, (batch_x, batch_y) in enumerate(tqdm(self.train_loader)):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                if self.args.dataset == 'CIFAR10':
                    # output, out_1000 = self.local_model(batch_x)
                    output = self.local_model(batch_x)
                elif self.args.dataset == 'MNIST':
                    output = self.local_model(batch_x)
                else:        
                    raise NotImplementedError(f'Unkown dataset {self.args.dataset}')
            
                loss = torch.nn.functional.cross_entropy(output, batch_y)
                # 计算梯度
                loss.backward()
                # 自更新
                optimizer.step()
        local_update = dict()
        for name, data in self.local_model.state_dict().items():
            local_update[name] = (data - global_model.state_dict()[name])
        # 缩放客户端更新
        if self.is_adversary and self.args.need_scale and attack_now:
            scale_upadate(self.args.weight_scale, local_update)
        
        print(f'# Epoch: {global_epoch} Client {self.client_id}  loss: {loss}\n')
        return local_update

def scale_upadate(weight_scale: int, local_update: Dict[str, torch.Tensor]):
    for name, value in local_update.items():
        value.mul_(weight_scale)
