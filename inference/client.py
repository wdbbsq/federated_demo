from base.client import BaseClient


class Client(BaseClient):
    def __init__(self, args, train_dataset, device, client_id=-1, is_adversary=False):
        super().__init__(args, train_dataset, device, client_id, is_adversary)

    def local_train(self, global_model, global_epoch, attack_now=False):
        # print('i would decrypt u')

        local_update = super().local_train(global_model, global_epoch, attack_now)
        # print('i would encrypt u')
        return local_update
