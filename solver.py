from bdb import Breakpoint
import torch
import torch.nn as nn
import torchvision
import os
import os.path as op
import sys
import time
from torch.utils.tensorboard import SummaryWriter

from utils.Logger import Logger

from utils.pyutils import dumpclean, set_seed, print_metrics, print_config
from model.build_model import build_model
from datasets.dataloader import get_loader
from metrics.losses.loss_function import FocalLoss, BCEFocalLoss
from metrics.evaluation_seg import get_accuray, get_sensitivity, get_specificity, get_precision, get_F1, get_DC, get_JS
from metrics.losses.SegLoss.losses_pytorch.dice_loss import DC_and_topk_loss, SoftDiceLoss, DC_and_CE_loss, FocalTversky_loss, IoULoss, SSLoss
from metrics.losses.SegLoss.losses_pytorch.focal_loss import FocalLoss

class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = 'checkpoint_{}.pth'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'pIoU', 'F1_socre', 'Dice']
    
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
        self.num_epochs = self.config.num_epochs # ?????????????????????
        self.epoch = 0 # ??????????????????????????????
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
        
        
        self.threshold_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        
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
        
        if loss_type == 'FocalLoss':
            self.loss_function = FocalLoss().cuda()
        elif loss_type == 'IoULoss':
            self.loss_function = IoULoss().cuda()
        elif loss_type == 'DC_and_CE_loss':
            self.loss_function = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {'weight': torch.tensor([0.01, 0.33, 0.66]), 'reduction': 'mean'}).cuda()
        elif loss_type == 'DC_and_topk_loss':
            self.loss_function = DC_and_topk_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {'k': 10}).cuda()
        elif loss_type == 'SSLoss':
            self.loss_function = SSLoss().cuda()
        elif loss_type == 'FocalTversky_loss':
            self.loss_function = FocalTversky_loss(tversky_kwargs={}, gamma=0.75).cuda()


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
        
        
    def validation(self, epoch):
        print("\nStart Validate Epoch-{}...".format(epoch))

        self.net.eval()

        metrics = {
            'acc':  {'{}'.format(k): 0. for k in self.threshold_list},
            'SE':   {'{}'.format(k): 0. for k in self.threshold_list},
            'SP':   {'{}'.format(k): 0. for k in self.threshold_list},
            'PC':   {'{}'.format(k): 0. for k in self.threshold_list},
            'F1':   {'{}'.format(k): 0. for k in self.threshold_list},
            'JS':   {'{}'.format(k): 0. for k in self.threshold_list},
            'DC':   {'{}'.format(k): 0. for k in self.threshold_list},
        }
        
        best_net_score = 0.
        best_threshold = 0.

        for batch_idx, (images, labels) in enumerate(self.val_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            seg_maps = self.net(images)
            seg_maps_prob = torch.sigmoid(seg_maps)
            

            for threshold in self.threshold_list:
                acc = metrics['acc'][f'{threshold}'] = metrics['acc'][f'{threshold}'] + get_accuray(seg_maps_prob, labels, threshold)
                DC = metrics['DC'][f'{threshold}']  = metrics['DC'][f'{threshold}'] + get_DC(seg_maps_prob, labels, threshold)
                F1 = metrics['F1'][f'{threshold}']  = metrics['F1'][f'{threshold}'] + get_F1(seg_maps_prob, labels, threshold)
                JS = metrics['JS'][f'{threshold}']  = metrics['JS'][f'{threshold}']+ get_JS(seg_maps_prob, labels, threshold)
                PC = metrics['PC'][f'{threshold}']  = metrics['PC'][f'{threshold}'] + get_precision(seg_maps_prob, labels, threshold)
                SE = metrics['SE'][f'{threshold}']  = metrics['SE'][f'{threshold}'] + get_sensitivity(seg_maps_prob, labels, threshold)
                SP = metrics['SP'][f'{threshold}']  = metrics['SP'][f'{threshold}'] + get_specificity(seg_maps_prob, labels, threshold)
                sum_score = DC + F1 + JS + PC + SE # ????????? acc ??? SP ????????????????????????
                if sum_score > best_net_score:
                    best_net_score, best_threshold = sum_score, threshold

            

        for metric in metrics.keys():
            for k in metrics[metric].keys():
                metrics[metric][k] = metrics[metric][k] / (batch_idx + 1)
        
        print_metrics(metrics, mode='val')

        return best_net_score, best_threshold
    

    def test(self):
        print("\nStart Testing...")
        print("Generating Visualization for Test Stage...")
        self.net.eval()
        self._load_checkpoint(checkpoint_type='best')
        self.net.eval()

        metrics = {
            'acc':  {'{}'.format(k): 0. for k in self.threshold_list},
            'SE':   {'{}'.format(k): 0. for k in self.threshold_list},
            'SP':   {'{}'.format(k): 0. for k in self.threshold_list},
            'PC':   {'{}'.format(k): 0. for k in self.threshold_list},
            'F1':   {'{}'.format(k): 0. for k in self.threshold_list},
            'JS':   {'{}'.format(k): 0. for k in self.threshold_list},
            'DC':   {'{}'.format(k): 0. for k in self.threshold_list},
        }

        best_net_score = 0.
        best_threshold = 0.

        for batch_idx, (images, labels) in enumerate(self.test_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            seg_maps = self.net(images)
            seg_maps_prob = torch.sigmoid(seg_maps)
            # seg_maps_prob = torch.softmax(seg_maps, dim=1)

            for threshold in self.threshold_list:
                acc = metrics['acc'][f'{threshold}'] = metrics['acc'][f'{threshold}'] + get_accuray(seg_maps_prob, labels, threshold)
                DC = metrics['DC'][f'{threshold}']  = metrics['DC'][f'{threshold}'] + get_DC(seg_maps_prob, labels, threshold)
                F1 = metrics['F1'][f'{threshold}']  = metrics['F1'][f'{threshold}'] + get_F1(seg_maps_prob, labels, threshold)
                JS = metrics['JS'][f'{threshold}']  = metrics['JS'][f'{threshold}']+ get_JS(seg_maps_prob, labels, threshold)
                PC = metrics['PC'][f'{threshold}']  = metrics['PC'][f'{threshold}'] + get_precision(seg_maps_prob, labels, threshold)
                SE = metrics['SE'][f'{threshold}']  = metrics['SE'][f'{threshold}'] + get_sensitivity(seg_maps_prob, labels, threshold)
                SP = metrics['SP'][f'{threshold}']  = metrics['SP'][f'{threshold}'] + get_specificity(seg_maps_prob, labels, threshold)
                sum_score = DC + F1 + JS + PC + SE # ????????? acc ??? SP ????????????????????????
                if sum_score > best_net_score:
                    best_net_score, best_threshold = sum_score, threshold

            SR_current_path = op.join(self.test_vis_folder, f"{batch_idx}_SegResults.png")
            RS_current_path = op.join(self.test_vis_folder, f"{batch_idx}_Label_Vis.png")
            IMG_current_path = op.join(self.test_vis_folder, f"{batch_idx}_Input.png")
            
            # ????????????????????????
            torchvision.utils.save_image(images.data.cpu(), IMG_current_path)

            # ??????????????????
            seg_results = seg_maps_prob.argmax(dim=1)
            rgb = torch.tensor([
                [0., 0., 0.], # channel one, class - background black
                [60/255, 180/255, 75/255], # channel two, class - olf green
                [250/255, 190/255, 212/255], # channel three, class - do pink
            ])
            seg_png = torch.zeros(seg_maps.size(0), 3, seg_maps.size(2), seg_maps.size(3))
            
            for batch in range(seg_results.size(0)):
                for h in range(seg_results.size(1)):
                    for w in range(seg_results.size(2)):
                        seg_png[batch, :, h, w] = rgb[seg_results[batch, h, w]]
            
            torchvision.utils.save_image(seg_png.data.cpu(), SR_current_path)
            
            # ?????????label
            
            torchvision.utils.save_image(labels.data.cpu(), RS_current_path) # ???????????? # ??????do # ??????olf
            '''
            save_results = torch.zeros(seg_maps.size(0), 3, seg_maps.size(2), seg_maps.size(3))
            RS = seg_maps_prob.clone()
            RS[RS > best_threshold] = 1
            RS[RS != 1] = 0
            save_results[:, 0, :, :] = labels[:, 0, :, :].data.cpu()
            save_results[:, 1, :, :] = RS[:, 0, :, :].data.cpu()
            torchvision.utils.save_image((save_results/save_results.max()).data.cpu(), RS_current_path)
            '''
        for metric in metrics.keys():
            for k in metrics[metric].keys():
                metrics[metric][k] = metrics[metric][k] / (batch_idx + 1)
        
        print_metrics(metrics, mode='test')

        
        print("Use threshold {}".format(best_threshold))


         
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
    
        best_net_score = -2. # Val and save model
        best_threshold = 0.9
        best_score_for_ReduceLROnPlateau = 0. # max mode for ReduceLROnPlateau lr_Scheduler
        
        epoch = self.epoch + 1
        
        print("\nStart Trainig... [Time]: {}".format(time.ctime()))
        
        while epoch <= self.num_epochs:
        
            print("\n\nStart Training Epoch-{} ...".format(epoch))
            
            self.net.train(True)
            
            epoch_loss = 0.
            
            metrics = {
                'acc':  {'{}'.format(k): 0. for k in self.threshold_list},
                'SE':   {'{}'.format(k): 0. for k in self.threshold_list},
                'SP':   {'{}'.format(k): 0. for k in self.threshold_list},
                'PC':   {'{}'.format(k): 0. for k in self.threshold_list},
                'F1':   {'{}'.format(k): 0. for k in self.threshold_list},
                'JS':   {'{}'.format(k): 0. for k in self.threshold_list},
                'DC':   {'{}'.format(k): 0. for k in self.threshold_list},
            }
    
        
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                images = images.to(self.device)
                labels = labels.to(self.device)

                # print(images.shape, labels.shape)
                # labels_flat = labels.view(labels.size(0), -1) # ????????????????????????loss
            
                seg_maps = self.net(images)
                
                seg_maps_prob = torch.sigmoid(seg_maps)
                # print(seg_maps.max(), seg_maps.min())
                # seg_probs = torch.sigmoid(seg_maps)
                # seg_flat = seg_probs.view(seg_probs.size(0), -1)
                # seg_flat = seg_maps.view(seg_maps.size(0), -1)
                
                loss = self.loss_function(seg_maps, labels)
                epoch_loss += loss.item()
                loss.backward()

                epoch_len = len(self.train_loader.dataset) // self.train_loader.batch_size
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + batch_idx)

                # Gradient Clipping
                if epoch < self.Gradient_Clip_Epoch:
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.5, norm_type=2)

                self.optimizer.step()
                
                for threshold in self.threshold_list:
                    acc = metrics['acc'][f'{threshold}'] = metrics['acc'][f'{threshold}'] + get_accuray(seg_maps_prob, labels, threshold)
                    DC = metrics['DC'][f'{threshold}']  = metrics['DC'][f'{threshold}'] + get_DC(seg_maps_prob, labels, threshold)
                    F1 = metrics['F1'][f'{threshold}']  = metrics['F1'][f'{threshold}'] + get_F1(seg_maps_prob, labels, threshold)
                    JS = metrics['JS'][f'{threshold}']  = metrics['JS'][f'{threshold}']+ get_JS(seg_maps_prob, labels, threshold)
                    PC = metrics['PC'][f'{threshold}']  = metrics['PC'][f'{threshold}'] + get_precision(seg_maps_prob, labels, threshold)
                    SE = metrics['SE'][f'{threshold}']  = metrics['SE'][f'{threshold}'] + get_sensitivity(seg_maps_prob, labels, threshold)
                    SP = metrics['SP'][f'{threshold}']  = metrics['SP'][f'{threshold}'] + get_specificity(seg_maps_prob, labels, threshold)
                    sum_score = DC + F1 + JS + PC + SE # ????????? acc ??? SP ????????????????????????
                    if sum_score > best_score_for_ReduceLROnPlateau:
                        best_score_for_ReduceLROnPlateau, best_threshold = sum_score, threshold
            
            for metric in metrics.keys():
                for k in metrics[metric].keys():
                    metrics[metric][k] = metrics[metric][k] / (batch_idx + 1)
                    
            
            writer.add_scalar('epoch_loss', epoch_loss, epoch)
            writer.add_scalar('train-acc', metrics['acc'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-DC', metrics['DC'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-F1', metrics['F1'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-JS', metrics['JS'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-PC', metrics['PC'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-SE', metrics['SE'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-SP', metrics['SP'][f'{best_threshold}'], epoch)
            writer.add_scalar('train-NetScore', best_score_for_ReduceLROnPlateau/(batch_idx + 1), epoch)


            # Print the log info 
            print("[Loss]: {:.4f}, [Net_Score]: {:.4f}, [Threshold]: {}".format(epoch_loss, best_score_for_ReduceLROnPlateau/(batch_idx + 1), best_threshold))
            print_metrics(metrics, mode='train')
            
            # Decay Learning Rate
            if self.config.lr_Scheduler != 'ReduceLROnPlateau':
                self.lr_Scheduler.step()
            else:
                self.lr_Scheduler.step(best_score_for_ReduceLROnPlateau)
            
            # Save Checkpoint
            self.save_checkpoint(epoch)
            
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
                    # ??????????????????????????????????????????????????????threshold?????????
                    
                    
            self.epoch = self.epoch + 1
            epoch = epoch + 1          
        
            # raise Breakpoint("DEBUG")

        writer.close()
        
        net_score, threshold = self.validation(epoch-1)
                    
        self.test()      
        
        print("\nFinish Training...[Time]: {}".format(time.ctime()))
                
        
        