from base.client import BaseClient


class Client(BaseClient):
    def __init__(self, args, train_dataset, device, client_id=-1, is_adversary=False):
        super(Client, self).__init__(args, train_dataset, device, client_id, is_adversary)
