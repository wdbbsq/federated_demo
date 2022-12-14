import argparse, json
import datetime
import os
import logging
import torch, random
import csv

from server import *
from client import *
import models, datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []

    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    with open("res/train.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "acc", "loss"])

    print("\n\n")
    for e in range(conf["global_epochs"]):

        candidates = random.sample(clients, conf["k"])

        # 客户端上传的模型更新
        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff = c.local_train(server.global_model)

            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()
        with open("res/train.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([e, acc, loss])

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    csvfile.close()




