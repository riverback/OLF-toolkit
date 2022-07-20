"""
Import code from PyTorch 
https://github.com/pytorch/vision/blob/e75a333782bb5d5ffdf5355e766eb5937fc6697c/torchvision/models/segmentation/_utils.py#L10

"""
from collections import OrderedDict
from typing import Optional
from torch import Tensor, nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]

        # backbone features
        features = self.backbone(x)


        result = OrderedDict()
        # if class_number > 1
        x = features['out']
        x = self.classifier(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        result['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x

        if self.aux_classifier is not None:
            return result
        else:
            return result['out']


