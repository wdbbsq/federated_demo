import time
import torch
from tqdm import tqdm

from base.server import BaseServer


class Server:
    def __init__(self, args):
        self.args = args
        self.accumulator = dict()
        self.rate = float(1 / args.k_workers)

    def accumulate_update(self, update):
        """
        累加加密参数
        """
        total_time = 0
        for name, data in update.items():
            if self.accumulator.__contains__(name):
                t1 = time.time()
                new_arr = [(self.accumulator[name][i] + data[i]) for i in range(len(data))]
                self.accumulator[name] = new_arr
                t2 = time.time()
                total_time += t2 - t1
            else:
                self.accumulator[name] = data
        print(f'Enc add in {total_time} \n')

    def model_aggregate(self):
        """
        计算平均
        """
        enc_model = dict()
        t1 = time.time()
        for name, data in self.accumulator.items():
            enc_model[name] = [(i * self.rate) for i in data]
        t2 = time.time()
        print(f'enc times with const in {t2 - t1}')
        return enc_model
