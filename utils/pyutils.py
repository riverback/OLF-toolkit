import random
import torch
import numpy as np
import os
from torch.backends import cudnn

def dumpclean(obj: dict):
    """To print a dictionary line by line"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print(k)
                dumpclean(v)
            else:
                print("{}: {}".format(k, v))
    elif isinstance(obj, list):
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print(v)
    else:
        print(obj)
        
def print_config(config):
    print("Config Information:")
    config_list = config._get_kwargs()
    for arg in config_list:
        print(arg)
                
                
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True


def print_metrics(metrics: dict, mode='train'):
    
    best_metrics = {k: ['', 0.] for k in metrics.keys()}
    
    for metric in metrics.keys():
        print("{}: (threshold, {})".format(metric, metric))
        for k, v in metrics[metric].items():
            print("({} {:.4f})".format(k, v), end=' ')
            if v > best_metrics[metric][1]:
                best_metrics[metric][0] = f'threshold-{k}'
                best_metrics[metric][1] = v
        print()
                
    if mode != 'train':
        print("\nBest metrics:")
        for k, v in best_metrics.items():
            print("{}: {:.4f} use {}".format(k, v[1], v[0]))