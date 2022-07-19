from datasets.OLFDataset import OLF_SEG_Dataset
from torch.utils import data


def get_loader(config):

    OLF_SEG_DATASET = OLF_SEG_Dataset(config)
    
    train_dataloader = data.DataLoader(
        dataset=OLF_SEG_DATASET.train_Dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_dataloader = data.DataLoader(
        dataset=OLF_SEG_DATASET.val_Dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_dataloader = data.DataLoader(
        dataset=OLF_SEG_DATASET.test_Dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    
    return train_dataloader, val_dataloader, test_dataloader
