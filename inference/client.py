import time
from copy import deepcopy

import numpy as np
import phe as paillier
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.client import BaseClient
from models import get_model
from utils.eval import model_evaluation
from utils.serialization import save_as_file


class Client(BaseClient):
    def __init__(self, args, train_dataset, device, client_id=-1, is_adversary=False):
        super(Client, self).__init__(args, train_dataset, device, client_id, is_adversary)
        self.private_key = None
        self.public_key = None
        self.model_shape = dict()
        self.eval_dataloader = None
        # 用于验证全局模型的性能
        self.global_model = None
        self.is_leader = False

    def init_params(self, eval_dataset):
        """
        领导客户端初始化Paillier密钥，全局模型
        """
        self.is_leader = True
        self.eval_dataloader = DataLoader(eval_dataset,
                                          batch_size=self.args.batch_size,
                                          shuffle=True)
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        global_model = get_model(self.args.model_name,
                                 self.device,
                                 input_channels=self.args.input_channels,
                                 output_num=self.args.nb_classes)
        self.save_model_shape(global_model)
        enc_param = self.encrypt_model(global_model)
        return self.private_key, self.public_key, enc_param, self.model_shape

    def save_secret(self, private_key, public_key, model_shape):
        """
        非领导节点执行 保存密钥以及模型结构
        """
        self.private_key = private_key
        self.public_key = public_key
        self.model_shape = deepcopy(model_shape)

    def save_model_shape(self, model):
        for name, data in model.state_dict().items():
            params = data.cpu().numpy()
            self.model_shape[name] = params.shape

    def encrypt_model(self, model):
        enc_dict = dict()
        t1 = time.time()
        for name, data in model.state_dict().items():
            params = data.cpu().numpy()
            enc_list = []
            for x in np.nditer(params):
                # 不防御就不加密
                x = self.public_key.encrypt(float(x)) if self.args.defense else x
                enc_list.append(x)
            enc_dict[name] = enc_list
        t2 = time.time()
        print(f'Enc params in {t2 - t1}')
        return enc_dict

    def get_local_model(self, global_model):
        """
        重写父类方法，解密全局模型
        """
        init_model = self.get_init_model()
        t1 = time.time()
        for name, data in init_model.state_dict().items():
            dec_list = []
            for x in global_model[name]:
                x = self.private_key.decrypt(x) if self.args.defense else x
                dec_list.append(x)
            # reshape vector
            dec_params = np.asarray(dec_list, dtype=np.float32).reshape(self.model_shape[name])
            # cover model params
            init_model.state_dict()[name].copy_(torch.from_numpy(dec_params))
        t2 = time.time()
        print(f'Dec params in {t2 - t1} \n')
        # 领导客户端每次训练都会保存全局模型用于评估
        if self.is_leader:
            del self.global_model
            self.global_model = init_model
        return init_model

    def calc_update(self, global_model, local_model, global_epoch, attack_now):
        """
        重写父类方法，加密本地更新
        """
        return self.encrypt_model(local_model)

    def eval_model(self, device, global_epoch, file_path):
        return model_evaluation(self.global_model, self.eval_dataloader, device, file_path,
                                global_epoch == self.args.global_epochs - 1)

