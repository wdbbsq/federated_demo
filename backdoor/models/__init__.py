import torch
from torchvision import models
from .badnet import BadNet


def get_model(name="vgg16", device=torch.device('cpu'), pretrained=True, input_channels=0, output_num=0):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
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
        model = BadNet(input_channels, output_num).to(device)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
