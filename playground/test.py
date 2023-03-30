import numpy as np
from utils import init_model

# public_key, private_key = paillier.generate_paillier_keypair()
net = init_model('resnet18')
# t = np.dtype([EncryptedNumber])
model_dict = dict()
# 加密参数
for name, data in net.state_dict().items():
    datalist = data.cpu().numpy()
    shape = datalist.shape
    arr = datalist.reshape(1, -1)[0].tolist()
    print(datalist)
    # 还原
    np.zeros(9408).reshape(datalist.shape)
    # values = np.ndarray(dtype=t)
    # for x in np.nditer(datalist):
    #     enc_x = public_key.encrypt(x.data)
