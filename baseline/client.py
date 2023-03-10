import torch
from tqdm import tqdm

from utils import init_model 
from torch.utils.data import DataLoader


class Client:

    def __init__(self, args, train_dataset, id=-1):
        self.args = args
        self.client_id = id
        self.local_model = init_model(self.args.model_name)
        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.args.total_workers)
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=args.batch_size,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                           train_indices)
                                       )

    def local_train(self, global_model, global_epoch):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum)
        training_loss = 0
        self.local_model.train()
        for e in range(self.args.local_epochs):
            for batch_id, batch in enumerate(tqdm(self.train_loader)):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                # 计算梯度
                loss.backward()
                # 自更新
                optimizer.step()
                training_loss += loss
        local_update = dict()
        for name, data in self.local_model.state_dict().items():
            local_update[name] = (data - global_model.state_dict()[name])
        training_loss /= len(self.train_loader)
        print(f'# Epoch: {global_epoch} Client {self.client_id} loss: {training_loss} \n')
        return local_update
