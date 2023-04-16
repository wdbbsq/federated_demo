from itertools import combinations

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from backdoor.defense.clip import compute_norms
from base.server import BaseServer
from utils.gradient import scale_update, get_vector, calc_vector_dist
from utils.serialization import save_as_file

MAX_CENTER_DIST = 1


class Server(BaseServer):
    def __init__(self, args, clean_eval_dataset, poisoned_eval_dataset, device):
        super(Server, self).__init__(args, clean_eval_dataset, device)
        self.defense = args.defense
        self.poisoned_dataloader = DataLoader(poisoned_eval_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)
        self.local_updates = []

    def preparation(self):
        super(Server, self).preparation()
        # 重置客户端更新记录
        self.local_updates = []

    def sum_update(self, client_update, client_id=-1):
        """
        重写父类方法，进行防御时记录更新
        """
        if self.defense:
            self.local_updates.append({
                'id': client_id,
                'local_update': client_update
            })
        else:
            super(Server, self).sum_update(client_update)

    def apply_defense(self, layer_name, k_workers, client_ids_map):
        """
        进行防御
        """
        # 聚类分析
        evil_list = []
        if self.args.cluster:
            evil_list = self.boot_cluster(layer_name, k_workers, client_ids_map)
        # 剪枝
        if self.args.clip:
            self.boot_clip(layer_name)
        # 累加客户端更新
        for update in self.local_updates:
            if update['id'] not in evil_list:
                super(Server, self).sum_update(update['local_update'])

    def boot_clip(self, layer_name):
        """
        枝的剪
        """
        norm_list, median_norm = compute_norms(self.global_model.state_dict(),
                                               self.local_updates, layer_name)
        for idx, update in enumerate(self.local_updates):
            scale_update(min(1, median_norm / norm_list[idx]), update['local_update'])

    def boot_cluster(self, layer_name, k_workers, client_ids_map):
        """
        :return: 恶意客户端ids
        """
        # 计算余弦相似度
        cos_list = np.zeros([k_workers, k_workers])
        id_seq_map, seq_id_map = client_ids_map['id_seq_map'], client_ids_map['seq_id_map']
        for i, j in list(combinations(self.local_updates, 2)):
            cos = cosine_similarity(get_vector(i['local_update'], layer_name),
                                    get_vector(j['local_update'], layer_name))[0][0]
            x, y = id_seq_map.get(i['id']), id_seq_map.get(j['id'])
            cos_list[x][y] = cos
            cos_list[y][x] = cos
        for i in range(k_workers):
            cos_list[i][i] = 1

        # 保存余弦相似度矩阵
        # if self.args.need_serialization:
        #     save_as_file({
        #         'cos_list': cos_list,
        #         'client_ids_map': client_ids_map
        #     }, f'{LOG_PREFIX}/{epoch}_cos_numpy')

        # kmeans
        clf = KMeans(n_clusters=2)
        clf.fit(cos_list)
        centers = clf.cluster_centers_
        dist = calc_vector_dist(centers[0].reshape(1, -1),
                                centers[1].reshape(1, -1))
        evil_list = []
        # 聚类中心大于阈值，才认为有恶意客户端
        if dist >= MAX_CENTER_DIST:
            labels = clf.labels_
            class_0, class_1 = [], []
            for idx, label in enumerate(labels):
                if label == 0:
                    class_0.append(idx)
                else:
                    class_1.append(idx)
            # 节点少的作为恶意客户端
            class_0 = class_1 if len(class_0) > len(class_1) else class_0
            for seq in class_0:
                evil_list.append(seq_id_map.get(seq))
        return evil_list

    def evaluate_backdoor(self, device, epoch, file_path):
        mta = self.eval_model(self.eval_dataloader, device, epoch, file_path)
        bta = self.eval_model(self.poisoned_dataloader, device, epoch, file_path)
        return {
            'clean_acc': mta['acc'], 'clean_loss': mta['loss'],
            'poisoned_acc': bta['acc'], 'poisoned_loss': bta['loss'],
        }

