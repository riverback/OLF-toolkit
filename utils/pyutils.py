import random
import torch
import numpy as np

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
                
                
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def print_metrics(metrics: dict, mode='train'):
    
    best_metrics = {k: ['', 0.] for k in metrics.keys()}
    
    for metric in metrics.keys():
        print("{}: (threshold, {})".format(metric, metric))
        for k, v in metric:
            print("({} {:.4f})".format(k, v), end=' ')
            if v > best_metrics[metrics][1]:
                best_metrics[metrics][0] = f'threshold-{k}'
                best_metrics[metrics][1] = v
                
    if mode != 'train':
        print("\nBest metrics:")
        for k, v in best_metrics:
            print("{}: {:.4f} use threshold {}".format(k, v[1], v[0]))