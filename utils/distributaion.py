import numpy
from datasets import dataset


def distribute_batches_equally(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :param num_workers: number of workers
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset


def convert_distributed_data_into_numpy(distributed_dataset):
    """
    Converts a distributed dataset (returned by a data distribution method) from Tensors into numpy arrays.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    """
    converted_distributed_dataset = []

    for worker_idx in range(len(distributed_dataset)):
        worker_training_data = distributed_dataset[worker_idx]

        X_ = numpy.array([tensor.numpy() for batch in worker_training_data for tensor in batch[0]])
        Y_ = numpy.array([tensor.numpy() for batch in worker_training_data for tensor in batch[1]])

        converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset


def generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size):
    """
    Generate data loaders from a distributed dataset.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param batch_size: batch size for data loader
    :type batch_size: int
    """
    data_loaders = []
    for worker_training_data in distributed_dataset:
        data_loaders.append(
            dataset.get_data_loader_from_data(batch_size, worker_training_data[0], worker_training_data[1],
                                              shuffle=True))

    return data_loaders
