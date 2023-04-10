import torch
from tqdm import tqdm

from base.server import BaseServer


class Server:
    def __init__(self, args):
        self.args = args
        self.accumulator = dict()
        self.rate = float(1 / args.k_workers)

    def accumulate_update(self, update):
        for name, data in update.items():
            if self.accumulator.__contains__(name):
                new_arr = [(self.accumulator[name][i] + data[i]) for i in range(len(data))]
                self.accumulator[name] = new_arr
            else:
                self.accumulator[name] = data

    def model_aggregate(self):
        enc_model = dict()
        for name, data in self.accumulator.items():
            enc_model[name] = [(i * self.rate) for i in data]
        return enc_model
