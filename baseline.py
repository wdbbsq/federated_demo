import argparse
import random

import yaml

from utils import datasets
from client import *
from server import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    parser.add_argument('-t', '--task', default="none")
    args = parser.parse_args()

    if args.task != "none":
        with open("configs/"+args.task+".yaml", encoding='utf-8') as f:
            task = yaml.load(f, Loader=yaml.FullLoader)
        if args.task == "backdoor":
            is_backdoor = True
        elif args.task == "poison":
            is_poison = True
        elif args.task == "inference":
            is_inference = True

    with open(args.conf, encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["dataset"])
    server = Server(conf, eval_datasets)
    clients = []
    for c in range(conf["total"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    save_result = conf["save_result"]

    # with open("res/train.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["epoch", "acc", "loss"])

    print("\n\n")
    for e in range(conf["global_epochs"]):
        candidates = random.sample(clients, conf["selected"])
        # 客户端上传的模型更新
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            local_update = c.local_train(server.global_model)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(local_update[name])

        server.model_aggregate(weight_accumulator)
        acc, loss = server.model_eval()
        # with open("res/train.csv", "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([e, acc, loss])

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    # csvfile.close()
