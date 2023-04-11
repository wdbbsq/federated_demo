import time

import numpy as np
from models import get_model
import phe as paillier

public_key, private_key = paillier.generate_paillier_keypair()
net = get_model('infer_net')
# t = np.dtype([EncryptedNumber])
model_dict = dict()

total_params = 0
enc_time = 0
dec_time = 0

# 加密参数
for name, data in net.state_dict().items():
    datalist = data.cpu().numpy()
    shape = datalist.shape

    length = 1
    for x in shape:
        length *= x
    enc_list = []
    total_params += length
    print(f'{name} has {length} params')

    # encrypt
    t1 = time.time()
    for x in np.nditer(datalist):
        x = float(x)
        enc_list.append(public_key.encrypt(x))
    t2 = time.time()
    enc_time += t2 - t1
    print(f'enc in {(t2 - t1)}')

    # 还原
    # enc_list = np.asarray(enc_list).reshape(datalist.shape)
    # enc_arr = enc_list.reshape(1, -1)[0].tolist

    dec_arr = []

    t3 = time.time()
    for x in enc_list:
        dec_arr.append(private_key.decrypt(x))
    t4 = time.time()
    dec_time += t4 - t3
    print(f'dec in {(t4 - t3)} \n')

print(f'total_params: {total_params}, enc_time: {enc_time}, dec_time: {dec_time}')
pass
