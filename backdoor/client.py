from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from models import get_model
from base.client import BaseClient
from utils.optimizer import get_optimizer
from utils.gradient import get_grads, scale_update
from utils.mgda import MGDASolver


class Client(BaseClient):
    def __init__(self, args, train_dataset, clean_train_set, device, client_id=-1, is_adversary=False):
        super(Client, self).__init__(args, train_dataset, device, client_id, is_adversary)
        # 用于计算主任务的loss
        self.clean_dataloader = DataLoader(clean_train_set,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           sampler=SubsetRandomSampler(self.train_indices)
                                           ) if is_adversary else None

    def calc_update(self, global_model, local_model, global_epoch, attack_now):
        local_update = super(Client, self).calc_update(global_model, local_model, global_epoch, attack_now)

        # 缩放客户端更新
        if self.is_adversary and self.args.need_scale and attack_now:
            scale_update(self.args.weight_scale, local_update)
        print(f'# Epoch: {global_epoch} / Client {self.client_id} done. \n')

        return local_update

    # def compute_backdoor_loss(self, params, model, criterion, inputs_back,
    #                               labels_back, grads=None):
    #     outputs = model(inputs_back)
    #     loss = criterion(outputs, labels_back)
    #
    #     if grads:
    #         grads = get_grads(params, model, loss)
    #
    #     return loss, grads

    # def local_train(self, global_model, global_epoch, attack_now=False):
    #
    #     local_model.train()
    #     clean_iter = enumerate(self.clean_dataloader) if self.is_adversary else None
    #
    #     for epoch in range(self.local_epochs):
    #         for batch_id, (batch_x, batch_y) in enumerate(tqdm(self.train_loader)):
    #             optimizer.zero_grad()
    #             batch_x = batch_x.to(self.device, non_blocking=True)
    #             batch_y = batch_y.to(self.device, non_blocking=True)
    #             output = local_model(batch_x)
    #
    #             clean_batch = next(clean_iter) if self.is_adversary else None
    #             if attack_now and self.is_adversary:
    #                 # mgda
    #                 grads, losses = {}, {}
    #                 clean_x = clean_batch[1][0].to(self.device, non_blocking=True)
    #                 clean_y = clean_batch[1][1].to(self.device, non_blocking=True)
    #                 clean_output = local_model(clean_x)
    #                 clean_loss = torch.nn.functional.cross_entropy(clean_output, clean_y)
    #                 clean_grad = get_grads(local_model, clean_loss)
    #                 grads['normal'] = clean_grad
    #                 losses['normal'] = clean_loss
    #
    #                 loss = torch.nn.functional.cross_entropy(output, batch_y)
    #                 grad = get_grads(local_model, loss)
    #                 grads['backdoor'] = grad
    #                 losses['backdoor'] = loss
    #
    #                 scale = MGDASolver.get_scales(grads, losses, 'loss+', ['backdoor', 'normal'])
    #
    #             else:
    #                 loss = torch.nn.functional.cross_entropy(output, batch_y)
    #             loss.backward()
    #             optimizer.step()
    #     local_update = dict()
    #     for name, data in local_model.state_dict().items():
    #         local_update[name] = (data - global_model.state_dict()[name])
    #
    #     # 缩放客户端更新
    #     if self.is_adversary and self.args.need_scale and attack_now:
    #         scale_update(self.args.weight_scale, local_update)
    #
    #     print(f'# Epoch: {global_epoch} / Client {self.client_id} done. loss: {loss.item()}\n')
    #     return local_update


