import torch.optim as optim


def get_optimizer(model, lr, momentum, optim_name='sgd'):
    if optim_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optim_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError()
