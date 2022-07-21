from bdb import Breakpoint
import torch
import torch.nn as nn
import torchvision
import os
import os.path as op
import sys
import time
from torch.utils.tensorboard import SummaryWriter

import torchmetrics
from torchmetrics.functional import accuracy, recall, precision, f1_score, specificity, dice_score

from utils.Logger import Logger
from utils.pyutils import set_seed, print_config
from model.build_model import build_model
from datasets.dataloader import get_loader



class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = 'checkpoint_{}.pth'
    _SPLITS = ('train', 'val', 'test')
    
    def __init__(self, config):
        # Get Config
        self.config = config
        
        # Experiment Log
        self.log_folder = op.join('experiments', self.config.experiment_name)
        if not op.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.log_path = op.join(self.log_folder, 'log.txt')
        sys.stdout = Logger(self.log_path)
        
        self.checkpoint_folder = op.join(self.log_folder, 'checkpoints')
        if not op.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        self.vis_folder = op.join(self.log_folder, 'vis')
        if not op.exists(self.vis_folder):
            os.makedirs(self.vis_folder)
        self.test_vis_folder = op.join(self.vis_folder, 'test')
        if not op.exists(self.test_vis_folder):
            os.makedirs(self.test_vis_folder)
        
        
        # Print Config
        print("Experiment: {}\nTime: {}".format(self.config.experiment_name, time.ctime()))
        print_config(self.config)

        # Set SummaryWriter
        self.summary_path = op.join(self.log_folder, 'summary')
        if not op.exists(self.summary_path):
            os.makedirs(self.summary_path)
        
        
        set_seed(self.config.seed)
        
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.cuda_idx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.lr = self.config.lr
        self.num_epochs = self.config.num_epochs # 总共的训练轮数
        self.epoch = 0 # 当前已经训练完的轮数
        self.num_epochs_decay = self.config.num_epochs_decay
        self.Gradient_Clip_Epoch = self.config.Gradient_Clip_Epoch
        
        # Set Model
        self.net = self._set_model()
        self.param_list = self.net.parameters()
        self.net.to(self.device)
        
        # Set Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.param_list, 
            lr=self.config.lr, 
            betas=[self.config.beta1, self.config.beta2], 
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # Set Learning Rate Scheduler
        self.lr_Scheduler = self._set_lr_Scheduler()
        
        # Set Loss Function
        self.loss_function = None
        self._set_loss_function(self.config.loss_type)
        
        # Set Dataloader
        self.train_loader, self.val_loader, self.test_loader = get_loader(self.config)
        
        
        
    def _set_model(self):
        net = build_model(self.config)
        print("\nset model as {}, type-{}".format(self.config.model, type(net)))
        return net
    
    
    def _set_lr_Scheduler(self):
        Scheduler = None
        if self.config.lr_Scheduler == 'ReduceLROnPlateau':
            Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10, verbose=True)
        elif self.config.lr_Scheduler == 'CosineAnnealingLR':
            Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, verbose=True)
        elif self.config.lr_Scheduler == 'ExponentialLR':
            Scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        elif self.config.lr_Scheduler == 'MultiStepLR':
            Scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,80], gamma=0.1, verbose=True)
        else:
            raise NotImplementedError("Only Support [ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, MultiStepLR]")
        
        return Scheduler


    def _set_loss_function(self, loss_type):

        if loss_type == 'CrossEntropy':
            self.loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.4, 0.4]).to(self.device))
        elif loss_type == 'FocalLoss':
            self.loss_function = ...
        else:
            raise NotImplementedError("unsupported loss function")

        print("set self.loss_function as: {}".format(type(self.loss_function)))

    def _torch_save_model(self, epoch, checkpoint_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    
    def save_checkpoint(self, epoch, tag=None):
        if tag is None:
            print("saving checkpoint for epoch {}\n".format(epoch))
            checkpoint_path = op.join(self.checkpoint_folder, 
                                    self._CHECKPOINT_NAME_TEMPLATE.format('%d'%(epoch)))
            self._torch_save_model(epoch, checkpoint_path)
            print("save checkpoint to {}\n".format(checkpoint_path))
        else:
            print("saving the best model")
            checkpoint_path = op.join(self.checkpoint_folder,
                                      self._CHECKPOINT_NAME_TEMPLATE.format(tag))
            self._torch_save_model(epoch, checkpoint_path)
            print("save the best model to {}\n".format(checkpoint_path))

        if self.epoch == self.num_epochs:
            print("save the last model")
            checkpoint_path = op.join(self.checkpoint_folder, 
                                  self._CHECKPOINT_NAME_TEMPLATE.format('last'))
            self._torch_save_model(epoch, checkpoint_path)
            print("save the last model to {}\n".format(checkpoint_path))
        
    
    def _load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = op.join(
            self.checkpoint_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type)
        )
        
        if op.isfile(checkpoint_path):
            print("load best model from {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        else:
            raise IOError("No checkpoint {}".format(checkpoint_path))
        
        
    def _reset_grad(self):
        """Zero the gradient buffers"""
        self.net.zero_grad()



    def train(self):

        writer = SummaryWriter(self.summary_path)

        if op.exists(op.join(self.checkpoint_folder,
                             self._CHECKPOINT_NAME_TEMPLATE.format('last'))):
            raise RuntimeError("Please change the experiment name for a new experiment")
        elif op.exists(op.join(self.checkpoint_folder,
                             self._CHECKPOINT_NAME_TEMPLATE.format('last'))):
            print("There is a 'last' checkpoint")
            print("Loading from the last checkpoint...")
            self.load_checkpoint('last')
            print("successfully load from the last checkpoint, continue training...")
        else:
            print("There is no checkpoint, train from scratch!")

        best_net_score = -2.
        best_score_for_ReduceLROnPlateau = 0.

        epoch = self.epoch + 1
        print("\nStart Trainig... [Time]: {}".format(time.ctime()))
        
        while epoch <= self.num_epochs:

            print("\n\nStart Training Epoch-{} ...[Time]: {}".format(epoch, time.ctime()))

            self.net.train(True)

            epoch_loss = 0.

            acc = 0.
            DC = 0.
            F1 = 0.
            PC = 0.
            SP = 0.
            Recall = 0.

            for batch_idx, (images, labels_encode, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                images = images.to(self.device)
                labels = labels.long().squeeze().to(self.device) # labels.shape = (B, 1) value in [0, C)
                labels_encode = labels_encode.long().to(self.device) # one-hot label.shape = (B, C, 1) value in [0,1]

                predicts = self.net(images)
                # predicts_prob = torch.softmax(predicts, dim=1) # 如果是ce loss就不用加
                predicts_output = predicts.argmax(dim=1)

                loss = self.loss_function(predicts, labels)
                epoch_loss += loss.item()
                loss.backward()

                epoch_len = len(self.train_loader.dataset) // self.train_loader.batch_size
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + batch_idx)

                # Gradient Clipping
                if epoch < self.Gradient_Clip_Epoch:
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.5, norm_type=2)

                self.optimizer.step()

                acc += accuracy(predicts_output, labels, num_classes=self.config.output_channels)
                DC += dice_score(predicts, labels)
                F1 += f1_score(predicts_output, labels, num_classes=self.config.output_channels)
                PC += precision(predicts_output, labels, num_classes=self.config.output_channels)
                SP += specificity(predicts_output, labels, num_classes=self.config.output_channels)
                Recall += recall(predicts_output, labels, num_classes=self.config.output_channels)


            acc /= (batch_idx + 1)
            DC /= (batch_idx + 1)
            F1 /= (batch_idx + 1)
            PC /= (batch_idx + 1)
            SP /= (batch_idx + 1)
            Recall /= (batch_idx + 1)
            sum_score = acc + DC + F1 + PC + SP + Recall
            if sum_score > best_score_for_ReduceLROnPlateau:
                best_score_for_ReduceLROnPlateau = sum_score

            writer.add_scalar('epoch_loss', epoch_loss, epoch)
            writer.add_scalar('train-acc', acc, epoch)
            writer.add_scalar('train-DC', DC, epoch)
            writer.add_scalar('train-F1', F1, epoch)
            writer.add_scalar('train-PC', PC, epoch)
            writer.add_scalar('train-SP', SP, epoch)
            writer.add_scalar('train-Recall', Recall, epoch)

            print('''
            [Loss]:         {:.4f}
            [Accuracy]:     {:.4f}
            [Precision]:    {:.4f}
            [Specificity]:  {:.4f}
            [Recall]:       {:.4f}
            [F1-score]:     {:.4f}
            [Dice-score]:   {:.4f}
            [Net-score]:    {:.4f}
            '''.format(epoch_loss, acc, PC, SP, Recall, F1, DC, best_score_for_ReduceLROnPlateau))

            if self.config.lr_Scheduler != 'ReduceLROnPlateau':
                    self.lr_Scheduler.step()
            else:
                self.lr_Scheduler.step(best_score_for_ReduceLROnPlateau)


            # Save Checkpoint
            self.save_checkpoint(epoch)
            '''
            # Validation
            if epoch % self.config.eval_frequency == 0:
                net_score, threshold = self.validation(epoch)
                writer.add_scalar("val_net_score", net_score, epoch)
                
                # Save the best model
                if net_score > best_net_score:
                    best_net_score = net_score
                    best_threshold = threshold
                    self.save_checkpoint(epoch, tag='best')
                    ###
                    # 这里要考虑一下保存的时候要记录最佳的threshold是多少
            '''        
                    
            self.epoch = self.epoch + 1
            epoch = epoch + 1          
        
            # raise Breakpoint("DEBUG")
            

        writer.close()
        '''
        net_score, threshold = self.validation(epoch-1)
                    
        self.test()      
        
        
        '''        
        print("\nFinish Training...[Time]: {}".format(time.ctime()))
