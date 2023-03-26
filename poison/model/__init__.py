import torch
import torch.nn as nn
from torchvision import models

from .net import Net


def get_model(name="vgg16", device=torch.device('cpu'), pretrained=True, input_channels=0, output_num=10):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # change input layer
        # the default number of input channel in the resnet is 3, but our images are 1 channel. So we have to change 3 to 1.
        # nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) <- default
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 

        # change fc layer
        # the number of classes in our dataset is 10. default is 1000.
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    elif name == 'badnets':
        model = Net(input_channels, output_num)

    return model.to(device)
