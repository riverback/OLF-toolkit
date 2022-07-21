from solver import Trainer as Trainer_seg
from classify_cam import Trainer as Trainer_cla
from utils.get_config import getConfig

def main(config):
    if config.task != 'Classify_XY':
        trainer = Trainer_seg(config)
    else:
        trainer = Trainer_cla(config)
    
    trainer.train()
    
    
if __name__ == '__main__':
    print("Good luck for you!")
    config = getConfig()
    main(config)
    