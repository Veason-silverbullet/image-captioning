import torch
import torchvision
import torch.nn as nn
from config import Config


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__() 
        self.cnn_encoder = config.cnn_encoder.lower()
        if self.cnn_encoder == 'resnet50' or self.cnn_encoder == 'resnet-50':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet101' or self.cnn_encoder == 'resnet-101':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet152' or self.cnn_encoder == 'resnet-152':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.resnet152(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'vgg19' or self.cnn_encoder == 'vgg-19':
            self.image_feature_channels = 512
            self.image_feature_dim = 14 * 14
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).children())[:-2][0][:-2])
        elif self.cnn_encoder == 'densenet121' or self.cnn_encoder == 'densenet-121':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.densenet121(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet161' or self.cnn_encoder == 'densenet-161':
            self.image_feature_channels = 2208
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.densenet161(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet169' or self.cnn_encoder == 'densenet-169':
            self.image_feature_channels = 1664
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.densenet169(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'googlenet' or self.cnn_encoder == 'inception-v1':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.CNN_encoder = nn.Sequential(*list(torchvision.models.googlenet(pretrained=True).children())[:-3])
        else:
            raise Exception('cnn encoder type not support: %s', cnn_encoder)
        config.image_feature_channels = self.image_feature_channels
        config.image_feature_dim = self.image_feature_dim
        for param in self.CNN_encoder.parameters():
            param.requires_grad = config.finetune_encoder if config.mode == 'train' else False

    def forward(self, images):
        image_feature = self.CNN_encoder(images).view([-1, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1) # [batch_size, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                      # [batch_size, channels]
        return image_feature, mean_image_feature

    def disable_BN(self):
        for module in self.CNN_encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
