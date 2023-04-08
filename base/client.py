import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from models import get_model
from utils.optimizer import get_optimizer


class BaseClient:
    def __init__(self, args, train_dataset, device, client_id, is_adversary):
        self.args = args
        self.device = device
        self.local_epochs = args.local_epochs
        self.client_id = client_id
        self.is_adversary = is_adversary

        all_range = list(range(len(train_dataset)))
        data_len = int(len(train_dataset) / args.total_workers)
        self.train_indices = all_range[client_id * data_len: (client_id + 1) * data_len]
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       sampler=SubsetRandomSampler(self.train_indices)
                                       )

    def preparation(self, global_model):
        """
        复制全局模型，并初始化优化器
        """
        local_model = get_model(self.args.model_name,
                                self.device,
                                input_channels=self.args.input_channels)
        for name, param in global_model.state_dict().items():
            local_model.state_dict()[name].copy_(param.clone())

        optimizer = get_optimizer(local_model,
                                  self.args.lr,
                                  self.args.momentum,
                                  self.args.optimizer)
        return local_model, optimizer

    def local_train(self, global_model, global_epoch, attack_now=False):

        local_model, optimizer = self.preparation(global_model)
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

        print(f'\n # Epoch: {global_epoch} Client {self.client_id} \n')
        return local_update
