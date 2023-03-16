import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from utils import init_model


class Client:

    def __init__(self, args, train_dataset, client_id=-1):
        self.args = args
        self.local_epochs = args.local_epochs
        self.local_model = init_model(args.model_name)
        self.client_id = client_id
        self.device = args.device

        all_range = list(range(len(train_dataset)))
        data_len = int(len(train_dataset) / args.total_workers)
        train_indices = all_range[client_id * data_len: (client_id + 1) * data_len]
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       sampler=SubsetRandomSampler(train_indices))

    def local_train(self, global_model, global_epoch):
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
                output = self.local_model(batch_x)
                loss = torch.nn.functional.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()
        local_update = dict()
        for name, data in self.local_model.state_dict().items():
            local_update[name] = (data - global_model.state_dict()[name])

        print(f'# Epoch: {global_epoch} Client {self.client_id}  loss: {loss.item()}\n')
        return local_update
