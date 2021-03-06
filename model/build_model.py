import torch.nn as nn
from model.Semantic_Segmentation.U_Net.unet import U_Net, AttU_Net
from model.Semantic_Segmentation.DeepLab.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101
from model.Semantic_Segmentation.TransUNet.model.transunet import build_transunet

from model.Backbone.vgg import vgg16_bn, vgg19_bn
from model.Backbone.resnet import resnet50, resnet101, resnet152, resnet34, resnet18

MODEL_LIST = ['U_Net', 'DeepLabV3_ResNet50', 'DeepLabV3_ResNet101', 'TransUNet', 'AttU_Net', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'VGG16', 'VGG19']

def build_model(config) -> nn.Module:
    if config.model not in MODEL_LIST:
        raise NotImplementedError("Now support {}".format(MODEL_LIST))
    
    model = None
    
    if config.model == 'U_Net':
        model = U_Net(config)
    elif config.model == 'AttU_Net':
        model = AttU_Net(config)
    elif config.model == 'DeepLabV3_ResNet50':
        model = deeplabv3_resnet50(config.image_channels, config.output_channels)
    elif config.model == 'DeepLabV3_ResNet101':
        model = deeplabv3_resnet101(config.image_channels, config.output_channels)
    elif config.model == 'TransUNet':
        model = build_transunet(config.vit_name, n_classes=config.output_channels)
    elif config.model == 'ResNet18':
        model = resnet18(config.image_channels, config.output_channels, config.down)
    elif config.model == 'ResNet34':
        model = resnet34(config.image_channels, config.output_channels, config.down)
    elif config.model == 'ResNet50':
        model = resnet50(config.image_channels, config.output_channels, config.down)
    elif config.model == 'ResNet101':
        model = resnet101(config.image_channels, config.output_channels, config.down)
    elif config.model == 'ResNet152':
        model = resnet152(config.image_channels, config.output_channels, config.down)
    elif config.model == 'VGG16':
        model = vgg16_bn(config.image_channels, config.output_channels, config.down)
    elif config.model == 'VGG19':
        model = vgg19_bn(config.image_channels, config.output_channels, config.down)
    else:
        ...
        
    return model
        
__all__ = [build_model]