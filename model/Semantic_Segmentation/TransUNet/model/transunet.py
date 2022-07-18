from model.Semantic_Segmentation.TransUNet.model.vit_seg_modeling import VisionTransformer as TransUNet
from model.Semantic_Segmentation.TransUNet.model.vit_seg_modeling import CONFIGS


import numpy as np

def build_transunet(vit_name=None, n_classes=1, n_skip=3, img_size=512, vit_patches_size=16):
    vit_name = vit_name
    if vit_name is None:
        vit_name = 'R50-ViT-B_16'
    config_TransUNet = CONFIGS[vit_name]
    config_TransUNet.n_classes = n_classes
    config_TransUNet.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_TransUNet.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = TransUNet(config_TransUNet, img_size=img_size, num_classes=n_classes)
    model.load_from(weights=np.load(config_TransUNet.pretrained_path))

    return model
