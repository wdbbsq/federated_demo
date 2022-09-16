# Environment

| Name | Version    |
|------|------------|
| OS | Windows 11 |
| CUDA | 11.3       |
| Python | 3.9        |
| PyTorch | 1.12.1     |



# config.json

1. `model_name`：模型名称 
2. `no_models`：客户端数量 
3. `type`：数据集信息 
4. `global_epochs`：全局迭代次数，即服务端与客户端的通信迭代次数 
5. `local_epochs`：本地模型训练迭代次数
6. `k`：每一轮迭代时，服务端会从所有客户端中挑选k个客户端参与训练。
7. `batch_size`：本地训练每一轮的样本数 
8. `lr`：学习率
9. `momentum`，`lambda`：本地模型的超参数设置