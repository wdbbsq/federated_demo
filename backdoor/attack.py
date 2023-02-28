import random

import torch


def poison_train_dataset(conf, train_dataset):
    """
    生成中毒数据集
    :param train_dataset:
    :param conf:
    :return:
    """
    #
    # return [(self.train_dataset[self.params['poison_image_id']][0],
    # torch.IntTensor(self.params['poison_label_swap']))]
    cifar_classes = {}
    for ind, x in enumerate(train_dataset):
        _, label = x
        if ind in conf.params['poison_images'] or ind in conf.params['poison_images_test']:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    indices = list()
    # create array that starts with poisoned images

    # create candidates:
    # range_no_id = cifar_classes[1]
    # range_no_id.extend(cifar_classes[1])

    # 剔除数据集中的后门样本
    range_no_id = list(range(50000))
    for image in conf.params['poison_images'] + conf.params['poison_images_test']:
        if image in range_no_id:
            range_no_id.remove(image)

    # add random images to other parts of the batch
    for batches in range(0, conf.params['size_of_secret_dataset']):
        range_iter = random.sample(range_no_id, conf.params['batch_size'])
        # range_iter[0] = self.params['poison_images'][0]
        indices.extend(range_iter)
        # range_iter = random.sample(range_no_id,
        #            self.params['batch_size']
        #                -len(self.params['poison_images'])*self.params['poisoning_per_batch'])
        # for i in range(0, self.params['poisoning_per_batch']):
        #     indices.extend(self.params['poison_images'])
        # indices.extend(range_iter)
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=conf.params['batch_size'],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
                                       )


def poison_test_dataset(conf, train_dataset):
    #
    # return [(self.train_dataset[self.params['poison_image_id']][0],
    # torch.IntTensor(self.params['poison_label_swap']))]
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=conf.params['batch_size'],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                           range(1000))
                                       )
