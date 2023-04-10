import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from models import get_model
from base.client import BaseClient

class Client(BaseClient):
    def __init__(self, args, train_dataset, device, client_id):
        super().__init__(args, train_dataset, device, client_id, False)
