import os
from os.path import join as ospj
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import random
from copy import deepcopy

from datasets.seg_olf_slice_info import OLF_SLICE


class OLFDataset(data.Dataset):
    def __init__(self, data, mode, std=None, mean=None):
        super().__init__()
        
        self.mode = mode
        self.std = std
        self.mean = mean
        
        self.image_size = 512
        
        self.Data = deepcopy(data)
        
        
    def __getitem__(self, index):
        
        image, label = self.Data[index]
            
        image, label = Image.fromarray(image), Image.fromarray(label)
        
        Transform = []
        if self.mode == 'train' and random.random() < 0.3:
            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
        Transform.append(T.Resize((self.image_size, self.image_size)))
        Transform.append(T.ToTensor())
        if self.mode is not None and self.std is not None:
            Transform.append(T.Normalize(self.mean, self.std))
        Transform = T.Compose(Transform)
        
        image, label = Transform(image), Transform(label)
        
        return image, label
        
        
    def __len__(self):
        return len(self.Data)
    
    @staticmethod
    def npy2list(arr):
        l = []
        for n in range(arr.shape[0]):
            l.append(arr[n, :, :])
        return l
    
class OLF_SEG_Dataset(object):
    def __init__(self, config):
        super().__init__()
        
        # Config 
        self.config = config
        self.split_prob = {'train': 0.4, 'val': 0.6, 'test': 1.0}
        
        # Path
        self.root = config.olf_root
        self.GT_root = ospj(self.root, 'GT')
        self.RAW_root = ospj(self.root, 'RAW')
        
        self.GT_folder_paths = [ospj(self.GT_root, path) for path in  os.listdir(self.GT_root)] # 001 002 ...
        self.GT_folder_paths.sort()
        self.RAW_paths = [ospj(ospj(self.RAW_root, 'npydata'), path) for path in os.listdir(ospj(self.RAW_root, 'npydata'))]
        self.RAW_paths.sort()
        
        if self.config.task == 'olf-seg-only':
            self.GT_paths = [ospj(folder_path, 'Label_seg_olf.npy') for folder_path in self.GT_folder_paths]    
            
        self.DataPackage = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for i in range(len(self.RAW_paths)):
            GT = np.load(self.GT_paths[i])
            # if GT.max() != 1:
            #     raise ValueError("GT for seg-olf only should have max value = 1")
            DCM = np.load(self.RAW_paths[i])
            for n in range(*OLF_SLICE[i+1]):
                p = random.random()
                if p < self.split_prob['train']:
                    self.DataPackage['train'].append([DCM[n, :, :], GT[n, :, :]])
                elif p < self.split_prob['val']:
                    self.DataPackage['val'].append([DCM[n, :, :], GT[n, :, :]])
                else:
                    self.DataPackage['test'].append([DCM[n, :, :], GT[n, :, :]])
        
        self.__aug_train_set()
        
        self.train_Dataset = OLFDataset(data=self.DataPackage['train'], mode='train', std=config.std, mean=config.mean)
        self.val_Dataset = OLFDataset(data=self.DataPackage['val'], mode='val')
        self.test_Dataset = OLFDataset(data=self.DataPackage['test'], mode='test')
    
    def __aug_train_set(self):
        print("Data Augmentation")
        p_aug = self.config.aug_prob
        RotationDegree = [-1, 1, 2, 3]
        for idx in range(len(self.DataPackage['train'])):
            if random.random() < p_aug:
                ori_img, ori_label = self.DataPackage['train'][idx]
                # left/right flip
                self.DataPackage['train'].append([np.fliplr(ori_img), np.fliplr(ori_label)])
                # up/down flip
                self.DataPackage['train'].append([np.flipud(ori_img), np.flipud(ori_label)])
                # left,up / right,down
                self.DataPackage['train'].append([np.fliplr(np.flipud(ori_img)), np.fliplr(np.flipud(ori_label))])
                # Color Jetter
                self.DataPackage['train'].append([ori_img*self.config.contraster_factor+self.config.brighness_factor, ori_label])
                # Adding Noise
                self.DataPackage['train'].append([self.__add_noise(ori_img, self.config.seed), ori_label])
                # Rotation
                k = RotationDegree[random.randint(0, 3)]
                self.DataPackage['train'].append([np.rot90(ori_img, k), np.rot90(ori_label, k)])
                # Shift
                shift_range = random.randint(50, 100)
                self.DataPackage['train'].append([self.__shift_arr(ori_img, shift_range), self.__shift_arr(ori_label, shift_range)])
        
    
    @staticmethod
    def __add_noise(arr, seed):
        rng = np.random.default_rng(seed)
        noise = rng.random(arr.shape)
        arr += noise
        min_v, max_v = arr.min(), arr.max()
        arr = (arr - min_v) / (max_v - min_v)
        return arr
    
    @staticmethod
    def __shift_arr(arr, num):
        assert num >= 0
        result = np.zeros_like(arr)
        result[:, :-num] = arr[:, num:]
        return result
            
            
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--olf_root', type=str, default=r'', help='')
    parser.add_argument('--task', type=str, default='olf-seg-only', help='[olf-seg-only,]')
    parser.add_argument('--mode', type=str, default='train')

    config = parser.parse_args()
    
    d = OLFDataset(config)

    print('breakpoint')

    #data_folder = r'C:\ZhuangResearchCode\OLF_TASK\Data\RAW'
    #paths = os.listdir(data_folder)
    #print(paths[0])
    # dcm_data = np.array(ReadDcmSequencePydicom(data_folder))
    # print(dcm_data.max(), dcm_data.min())
    
    