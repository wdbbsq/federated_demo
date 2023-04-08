# import numpy as np
# from utils import init_model
#
# # public_key, private_key = paillier.generate_paillier_keypair()
# net = init_model('resnet18')
# # t = np.dtype([EncryptedNumber])
# model_dict = dict()
# # 加密参数
# for name, data in net.state_dict().items():
#     datalist = data.cpu().numpy()
#     shape = datalist.shape
#     arr = datalist.reshape(1, -1)[0].tolist()
#     print(datalist)
#     # 还原
#     np.zeros(9408).reshape(datalist.shape)
#     # values = np.ndarray(dtype=t)
#     # for x in np.nditer(datalist):
#     #     enc_x = public_key.encrypt(x.data)
import argparse

parser = argparse.ArgumentParser(description='description')

parser.add_argument('--loss', dest='base', default='mse',
                    help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', dest='base', default='sgd',
                    help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--global_epochs', dest='global', type=int, default=100,
                    help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', dest='global', type=int, default=64,
                    help='Batch size to split dataset, default: 64')
parser.add_argument('--test_batch_size', dest='global', type=int, default=64,
                    help='Batch size to split dataset, default: 64')
args = parser.parse_args()
ff = vars(args)
print(ff)
