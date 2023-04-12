from torch.utils.data import DataLoader

from backdoor.defense.clip import compute_norms
from base.server import BaseServer
from utils.gradient import scale_update


class Server(BaseServer):
    def __init__(self, args, clean_eval_dataset, poisoned_eval_dataset, device):
        super(Server, self).__init__(args, clean_eval_dataset, device)
        self.poisoned_dataloader = DataLoader(poisoned_eval_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    def apply_defense(self, layer_name, local_updates):
        norm_list, median_norm = compute_norms(self.global_model.state_dict(),
                                               local_updates, layer_name)
        for idx, update in enumerate(local_updates):
            scale_update(min(1, median_norm / norm_list[idx]), update['local_update'])

    def evaluate_backdoor(self, device, epoch, file_path):
        mta = self.eval_model(self.eval_dataloader, device, epoch, file_path)
        bta = self.eval_model(self.poisoned_dataloader, device, epoch, file_path)
        return {
            'clean_acc': mta['acc'], 'clean_loss': mta['loss'],
            'poisoned_acc': bta['acc'], 'poisoned_loss': bta['loss'],
        }

