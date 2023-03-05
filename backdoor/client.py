from tqdm import tqdm
import models

import torch


class Client:
    def __init__(self, args, train_dataset, client_id=-1, is_adversary=False):
        self.args = args
        self.local_model = models.get_model(args.model_name,
                                            args.device,
                                            input_channels=args.input_channels,
                                            output_num=args.nb_classes)
        self.client_id = client_id
        self.is_adversary = is_adversary

        all_range = list(range(len(train_dataset)))
        data_len = int(len(train_dataset) / args.total_num)
        train_indices = all_range[client_id * data_len: (client_id + 1) * data_len]
        # todo 每个客户端自己的验证集
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices)
                                                        )

    def local_train(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum)
        self.local_model.train()

        for epoch in range(self.args.local_epoch):
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
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        print(f'# Client {self.client_id}  loss: {loss}\n')
        return {
            'local_update': diff,
            'loss': loss
        }
