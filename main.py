from solver import Trainer
from utils.get_config import getConfig

def main(config):
    trainer = Trainer(config)
    trainer.train()
    
    
if __name__ == '__main__':
    print("Good luck for you!")
    config = getConfig()
    main(config)
    