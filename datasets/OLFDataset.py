import os
from os.path import join as ospj
from typing import List
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import random
from copy import deepcopy

from datasets.seg_olf_slice_info import OLF_SLICE
from datasets.dcmutils import normalize_minmax
from datasets.prepare_classify_and_cam import ReadDcmSequence_XY_Pydicom


class OLFDataset(data.Dataset):
    def __init__(self, data: List, mode, std=None, mean=None):
        super().__init__()
        
        self.mode = mode
        self.std = std
        self.mean = mean
        
        self.image_size = 512
        
        self.Data = deepcopy(data)
        
        if self.mode == 'train':
            for index in range(len(self.Data)):
                image, label = self.Data[index]
                image, label = torch.from_numpy(image), torch.from_numpy(label)
                image = image.unsqueeze(0).type(torch.FloatTensor)
                
                Transform = []
                Transform.append(T.Resize((self.image_size, self.image_size)))
                Transform = T.Compose(Transform)
                image, label = Transform(image), Transform(label)
                Transform = []
                if self.mode is not None and self.std is not None:
                    Transform.append(T.Normalize(self.mean, self.std))
                    Transform = T.Compose(Transform)
                    image= Transform(image)
                self.Data[index] = [image, label]
                
                
                image_, label_ = deepcopy(image), deepcopy(label)

                Transform = []
                RotationRange = random.randint(-10, 10)
                Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                Transform.append(T.Resize((self.image_size, self.image_size)))
                Transform = T.Compose(Transform)
                self.Data.append([Transform(image_), Transform(label_)])
        
        else:
            for index in range(len(self.Data)):
                image, label = self.Data[index]
                image, label = torch.from_numpy(image), torch.from_numpy(label)
                image = image.unsqueeze(0).type(torch.FloatTensor)
                
                Transform = []
                Transform.append(T.Resize((self.image_size, self.image_size)))
                Transform = T.Compose(Transform)
                image, label = Transform(image), Transform(label)
                Transform = []
                if self.mode is not None and self.std is not None:
                    Transform.append(T.Normalize(self.mean, self.std))
                    Transform = T.Compose(Transform)
                    image = Transform(image)
                self.Data[index] = [image, label]
                
        if self.__len__() % 2 != 0:
            self.Data = self.Data[:-1]

        print("{} Images in {} dataset".format(self.__len__(), self.mode))
        
        
    def __getitem__(self, index):
        
        image, label = self.Data[index]

        return image, label
        
        
    def __len__(self):
        return len(self.Data)
    
    @staticmethod
    def npy2D2list(arr):
        l = []
        for n in range(arr.shape[0]):
            l.append(arr[n, :, :])
        return l
    
class OLF_SEG_Dataset(object):
    def __init__(self, config):
        super().__init__()
        
        # Config 
        self.config = config
        self.split_prob = {'train': 0.34, 'val': 0.67, 'test': 1.0}
        
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
        elif self.config.task == 'seg-do+olf':
            self.GT_paths = [ospj(folder_path, 'Label_seg_olf_and_do.npy') for folder_path in self.GT_folder_paths][:2] # 001 002, 003还不知道第几节是DO
            self.RAW_paths = self.RAW_paths[:2] # 没有003
        
        print(f"GT: {self.GT_paths}\nRAW: {self.RAW_paths}")

        self.DataPackage = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for i in range(len(self.RAW_paths)): # 每次读取一整个文件夹的数据

            GT = np.load(self.GT_paths[i])

            DCM = np.load(self.RAW_paths[i])
            
            print(GT.shape, DCM.shape)

            if GT.max() > 1. and self.config.task == 'olf-seg-only':
                raise ValueError("GT for seg-olf only should have max value = 1")

            if DCM.max() > 1.:
                DCM = normalize_minmax(DCM)

            for n in range(*OLF_SLICE[i+1]): # 目前数据少,采用随机分配的方式划分,之后这里直接本地划分文件夹就可以了,目前这种方式有数据泄露的嫌疑
                p = random.random()
                if p < self.split_prob['train']:
                    self.DataPackage['train'].append([DCM[n], GT[n]])
                elif p < self.split_prob['val']:
                    self.DataPackage['val'].append([DCM[n], GT[n]])
                else:
                    self.DataPackage['test'].append([DCM[n], GT[n]])

        
        self.__aug_train_set() # 对训练集进行数据增广


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

                # Color Jetter
                self.DataPackage['train'].append([self.__random_color_jetter(ori_img), ori_label])
                # Adding Noise
                self.DataPackage['train'].append([self.__add_noise(ori_img, self.config.seed), ori_label])
                # Shift
                shift_range = random.randint(50, 100)
                self.DataPackage['train'].append([self.__shift_arr(ori_img, shift_range), self.__shift_arr(ori_label, shift_range)])
                
                # Rotation
                # left/right flip
                # up/down flip
                # left,up / right,down
                k = RotationDegree[random.randint(0, 3)]
                if len(ori_label.shape) == 2:
                    self.DataPackage['train'].append([np.rot90(ori_img, k, axes=(0, 1)), np.rot90(ori_label, k, axes=(0, 1))])
                    self.DataPackage['train'].append([np.flip(ori_img, 1), np.flip(ori_label, 1)])
                    self.DataPackage['train'].append([np.flip(ori_img, 0), np.flip(ori_label, 0)])
                    self.DataPackage['train'].append([np.flip(np.flip(ori_img, 0), 1), np.flip(np.flip(ori_label, 0), 1)])
                else:
                    self.DataPackage['train'].append([np.rot90(ori_img, k, axes=(0, 1)), np.rot90(ori_label, k, axes=(1, 2))])
                    self.DataPackage['train'].append([np.flip(ori_img, 1), np.flip(ori_label, 2)])
                    self.DataPackage['train'].append([np.flip(ori_img, 0), np.flip(ori_label, 1)])
                    self.DataPackage['train'].append([np.flip(np.flip(ori_img, 0), 1), np.flip(np.flip(ori_label, 1), 2)])


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
        if len(arr.shape) == 2:
            result[:, :-num] = arr[:, num:]
        elif len(arr.shape) == 3:
            result[:, :-num, :] = arr[:, num:, :]
        return result
    
    @staticmethod
    def __random_color_jetter(arr):
        brightness_factor = random.uniform(0.2, 0.4)
        contraster_factor = random.uniform(0.2, 0.6)
        return arr * contraster_factor + brightness_factor

class Classify_Dataset(data.Dataset):
    def __init__(self, mode, Data: List, std=None, mean=None):
        super().__init__()
        
        self.mode = mode
        self.std = std
        self.mean = mean
        
        self.image_size = 512
        
        self.Data = deepcopy(Data)

        if self.mode == 'train':
            for index in range(len(self.Data)):
                flag = False
                image, label_encode, label = self.Data[index]
                if label != [0]: 
                    flag = True
                image, label_encode, label = torch.tensor(image), torch.tensor(label_encode), torch.tensor(label)
                image = image.unsqueeze(0).type(torch.FloatTensor)                                

                Transform = []
                Transform.append(T.Resize((self.image_size, self.image_size)))
                Transform = T.Compose(Transform)
                image = Transform(image)
                Transform = []
                if self.mode is not None and self.std is not None:
                    Transform.append(T.Normalize(self.mean, self.std))
                    Transform = T.Compose(Transform)
                    image= Transform(image)
                self.Data[index] = [image, label_encode, label]
                
                if flag:
                    image_ = image.clone().detach()

                    Transform = []
                    RotationRange = random.randint(-10, 10)
                    Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                    Transform.append(T.Resize((self.image_size, self.image_size)))
                    Transform = T.Compose(Transform)
                    self.Data.append([Transform(image_), label_encode, label])

        else:
            for index in range(len(self.Data)):
                image, label_encode, label = self.Data[index]
                image, label_encode, label = torch.tensor(image), torch.tensor(label_encode), torch.tensor(label)
                image = image.unsqueeze(0).type(torch.FloatTensor)                                

                Transform = []
                Transform.append(T.Resize((self.image_size, self.image_size)))
                Transform = T.Compose(Transform)
                image = Transform(image)
                Transform = []
                if self.mode is not None and self.std is not None:
                    Transform.append(T.Normalize(self.mean, self.std))
                    Transform = T.Compose(Transform)
                    image= Transform(image)
                self.Data[index] = [image, label_encode, label]

        if len(self.Data) % 2 != 0:
            self.Data = self.Data[:-1]

        print(f"{self.mode} Dataset, {self.__len__()} Images")

    def __getitem__(self, index):
        image, label_encode, label = self.Data[index]

        return image, label_encode, label

    def __len__(self):
        return len(self.Data)
        

class Classify_XY_Dataset(object):
    def __init__(self, config):
        super().__init__()     

        # Config
        self.config = config

        self.root = config.olf_root
        self.GT_root = ospj(self.root, 'GT')
        self.RAW_root = ospj(self.root, 'RAW')

        self.GT_folder_paths = [ospj(self.GT_root, path) for path in  os.listdir(self.GT_root)] # 001 002 ...
        # self.GT_folder_paths.sort()
        self.GT_paths = [ospj(ospj(self.GT_root, '001'), 'Label_class3_xy.txt'), ospj(ospj(self.GT_root, '002'), 'Label_class3_xy.txt')]
        self.RAW_paths = [ospj(self.RAW_root, '001'), ospj(self.RAW_root, '002')]
        # self.RAW_paths.sort()


        if len(self.GT_paths) != len(self.RAW_paths):
            raise ValueError("GT size is not equal to RAW size")

        self.Data = []

        for i in range(len(self.GT_paths)):
            f = open(self.GT_paths[i], 'r')
            label_f = f.readlines()
            dcm_data = ReadDcmSequence_XY_Pydicom(self.RAW_paths[i])

            for index in range(dcm_data.shape[0]):
                label = int(label_f[index].strip('\n').split()[-1])
                label_encode = self.make_ont_hot(label)
                label = [label]
                dcm = dcm_data[index]
                self.Data.append([dcm, label_encode, label])

            f.close()

        random.shuffle(self.Data)


        data_cnt = len(self.Data)
        split_train, split_val = int(0.5*data_cnt), int(0.99*data_cnt)

        self.DataPackage = {
            'train': self.Data[:split_train],
            'val': self.Data[split_train:split_val],
            'test': self.Data[split_val:],
        }

        self._aug_train()

        self.train_Dataset = Classify_Dataset(mode='train', Data=self.DataPackage['train'], std=config.std, mean=config.mean)
        self.val_Dataset = Classify_Dataset(mode='val', Data=self.DataPackage['val'])
        self.test_Dataset = Classify_Dataset(mode='test', Data=self.DataPackage['test'])


    def _aug_train(self):
        print("Data Augmentation")
        p_aug = self.config.aug_prob
        RotationDegree = [-1, 1, 2, 3]
        for idx in range(len(self.DataPackage['train'])):
            if random.random() < p_aug:
                ori_img, label_encode, label = self.DataPackage['train'][idx]

                # Color Jetter
                self.DataPackage['train'].append([self.__random_color_jetter(ori_img), label_encode, label])
                # Adding Noise
                self.DataPackage['train'].append([self.__add_noise(ori_img, self.config.seed), label_encode, label])
                # Shift
                shift_range = random.randint(50, 100)
                self.DataPackage['train'].append([self.__shift_arr(ori_img, shift_range), label_encode, label])
                
                # Rotation
                # left/right flip
                # up/down flip
                # left,up / right,down
                k = RotationDegree[random.randint(0, 3)]

                self.DataPackage['train'].append([np.rot90(ori_img, k, axes=(0, 1)), label_encode, label])
                self.DataPackage['train'].append([np.flip(ori_img, 1), label_encode, label])
                self.DataPackage['train'].append([np.flip(ori_img, 0), label_encode, label])
                self.DataPackage['train'].append([np.flip(np.flip(ori_img, 0), 1), label_encode, label])


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
        if len(arr.shape) == 2:
            result[:, :-num] = arr[:, num:]
        elif len(arr.shape) == 3:
            result[:, :-num, :] = arr[:, num:, :]
        return result
    
    @staticmethod
    def __random_color_jetter(arr):
        brightness_factor = random.uniform(0.2, 0.4)
        contraster_factor = random.uniform(0.2, 0.6)
        return arr * contraster_factor + brightness_factor


    @staticmethod
    def make_ont_hot(label: int):
        """
        label:
            0 - healthy
            1 - olf
            2 - do
        """
        label_encoded = torch.zeros([3])
        label_encoded[label] = float(label)

        return label_encoded


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
    
    