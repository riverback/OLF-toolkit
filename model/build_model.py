import torch.nn as nn
from model.Semantic_Segmentation.U_Net.unet import U_Net
from model.Semantic_Segmentation.DeepLab.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101
from model.Semantic_Segmentation.TransUNet.model.transunet import build_transunet

MODEL_LIST = ['U_Net', 'DeepLabV3_ResNet50', 'DeepLabV3_ResNet101', 'TransUNet']

def build_model(config) -> nn.Module:
    if config.model not in MODEL_LIST:
        raise NotImplementedError("Now support {}".format(MODEL_LIST))
    
    model = None
    
    if config.model == 'U_Net':
        model = U_Net(config)
    elif config.model == 'DeepLabV3_ResNet50':
        model = deeplabv3_resnet50(config.image_channels, config.output_channels)
    elif config.model == 'DeepLabV3_ResNet101':
        model = deeplabv3_resnet101(config.image_channels, config.output_channels)
    elif config.model == 'TransUNet':
        model = build_transunet(config.vit_name, n_classes=config.output_channels)
    else:
        ...
        
    return model
        
__all__ = [build_model]