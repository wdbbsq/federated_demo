from tqdm import tqdm
from typing import Dict
from backdoor.model import get_model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Client:

    def __init__(self, args, train_dataset, device, client_id=-1, is_adversary=False):
        self.args = args
        self.device = device
        self.local_epochs = args.local_epochs
        self.client_id = client_id
        self.is_adversary = is_adversary

        all_range = list(range(len(train_dataset)))
        data_len = int(len(train_dataset) / args.total_workers)
        train_indices = all_range[client_id * data_len: (client_id + 1) * data_len]
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
                                       )

    def local_train(self, global_model, global_epoch, attack_now=False):
        local_model = get_model(self.args.model_name,
                                     self.device,
                                     input_channels=self.args.input_channels,
                                     output_num=self.args.nb_classes)
        for name, param in global_model.state_dict().items():
            local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(local_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum)
        local_model.train()

        for epoch in range(self.local_epochs):
            for batch_id, (batch_x, batch_y) in enumerate(tqdm(self.train_loader)):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = local_model(batch_x)
                loss = torch.nn.functional.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()
        local_update = dict()
        for name, data in local_model.state_dict().items():
            local_update[name] = (data - global_model.state_dict()[name])

        # 缩放客户端更新
        if self.is_adversary and self.args.need_scale and attack_now:
            scale_update(self.args.weight_scale, local_update)

        print(f'# Epoch: {global_epoch} Client {self.client_id}  loss: {loss.item()}\n')
        return local_update


def scale_update(weight_scale: int, local_update: Dict[str, torch.Tensor]):
    for name, value in local_update.items():
        value.mul_(weight_scale)
