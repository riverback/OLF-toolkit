from OLFDataset import OLFDataset
from torch.utils import data


def get_loader(config):
    
    dataset = OLFDataset(config)
    
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    
    return data_loader
