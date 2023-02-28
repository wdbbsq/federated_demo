
from tqdm import tqdm

import models
import torch


class Client:
    def __init__(self, conf, model, train_dataset, client_id=-1, is_adversary=False):
        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.is_adversary = is_adversary

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf["total"])
        train_indices = all_range[client_id * data_len: (client_id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
                                                        )

    def local_train(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(),
                                    lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in tqdm(enumerate(self.train_loader)):
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
            # print("Client %d : epoch %d done." % (self.client_id, e))
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        print("Client %d done." % self.client_id)
        return diff
