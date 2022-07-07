import torch.nn as nn
from model.unet import U_Net

MODEL_LIST = ['U_Net']

def build_model(config) -> nn.Module:
    if config.model not in MODEL_LIST:
        raise NotImplementedError("Now support {}".format(MODEL_LIST))
    
    model = None
    
    if config.model == 'U_Net':
        model = U_Net(config)
    elif config.model == '':
        ...
    else:
        ...
        
    return model
        
__all__ = [build_model]