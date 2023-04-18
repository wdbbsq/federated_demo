import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from tqdm import tqdm
from models import get_model
from poison.defense import get_clean_updates
from utils.serialization import save_as_file
from base.server import BaseServer


class Server(BaseServer):
    def __init__(self, args, eval_dataset, device):
        super(Server, self).__init__(args, eval_dataset, device)

    def apply_defense(self, client_ids_map, file_name, epoch):
        """
        进行防御
        """
        # 计算得到干净样本
        clean_nodes = get_clean_updates(self.local_updates, self.args.defense_method)
        # 累加客户端更新
        id_seq_map = client_ids_map['id_seq_map']
        for idx in clean_nodes:
            for name, params in self.global_model.state_dict().items():
                self.weight_accumulator[name].add_(self.local_updates[id_seq_map.get(idx)]['local_update'][name])
        # 保存中间结果
        if self.args.need_serialization:
            save_as_file(self.local_updates, f'{file_name}/{epoch}_dist')
