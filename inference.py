import argparse
import random

import yaml

from datasets import datasets
from inference.inference.inference_client import *
from baseline.server import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Inference')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open("configs/backdoor.yaml", encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["dataset"])


    server = Server(conf, eval_datasets)
    clients = []
    adversary_list = random.sample(range(conf["total"]), conf["adversary_num"])
    for i in range(conf["total"]):
        clients.append(Client(conf, server.global_model, train_datasets, i, i in adversary_list))

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
        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

