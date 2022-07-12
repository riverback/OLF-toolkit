import os
import argparse



def getConfig():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--cuda_idx', type=str, default='7',
                        help='multi-gpu: "1,2" ')
    parser.add_argument('--seed', type=int, default=10,
                        help='')
    
    # Path
    parser.add_argument('--olf_root', type=str, default=r'Data',
                        help='path for olf-data root')
    parser.add_argument('--experiment_name', type=str, default='DEBUG',
                        help='name for experiment log')
    
    # DataSet and DataLoader
    parser.add_argument('--task', type=str, default='olf-seg-only',
                        help='[olf-seg-only, ]')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--std', type=float, default=1.,
                        help='normalize when preparing data')
    parser.add_argument('--mean', type=float, default=0.,
                        help='normalize when preparing data')
    parser.add_argument('--aug_prob', type=float, default=0.3,
                        help='aug prob for train dataset')
    parser.add_argument('--contraster_factor', type=float, default=0.2,
                        help='Color Jetter factor')
    parser.add_argument('--brighness_factor', type=float, default=0.2,
                        help='Color Jetter factor')
    
    # Basic Model Hyperparameters
    parser.add_argument('--image_channels', type=int, default=1, 
                        help='olf ct data is 1-channel')
    parser.add_argument('--output_channels', type=int, default=1, 
                        help='olf-seg-only=1, olf-do-seg=2')
    
    # Traing Parameter Settings
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--num_epochs_decay', type=int, default=30,
                        help='start decay lr after these epochs')
    parser.add_argument('--Gradient_Clip_Epoch', type=int, default=10,
                        help='gradient clipping in fisrt several epochs')
    parser.add_argument('--lr_Scheduler', type=str, default='ExponentialLR',
                        help='[ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, MultiStepLR]')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam settings')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam settings')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Adam settings')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Adam settings')

    # Loss Function
    parser.add_argument('--loss_type', type=str, default='DC_and_topk_loss')
    
    # Val Settings
    parser.add_argument('--eval_frequency', type=int, default=2,
                        help='eval frequency')
    
    # Basic Model Setting
    parser.add_argument('--model', type=str, default='U_Net',
                        help='[U_Net, DeepLabV3_ResNet50, DeepLabV3_ResNet101]')
    
    # Architecture Hyperparameters


    
    config = parser.parse_args()
    
    return config