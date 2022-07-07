from datasets.OLFDataset import OLFDataset
from torch.utils import data


def get_loader(config):
    
    dataset = OLFDataset(config)
    
    if config.mode == 'train':
        shuffle_flag = True
    else:
        shuffle_flag = False
        
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle_flag,
        num_workers=config.num_workers
    )
    
    return data_loader
