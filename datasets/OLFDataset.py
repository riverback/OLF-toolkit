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


from datasets.seg_olf_slice_info import OLF_SLICE


class OLFDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        
        # Config
        self.config = config
        self.mode = config.mode # train val test

        # args for data augmentation and process
        self.image_size = 512 # CT数据通常为 512x512xZ 
        self.aug_prob = 0.5
        self.RotationDegree = [0, 90, 180, 270]
        if 'mean' not in config:
            self.mean = 0.
        else:
            self.mean = config.mean
        if 'std' not in config:
            self.std = 1.
        else:
            self.std = config.std
        
        # Path
        self.root = config.olf_root
        self.GT_root = ospj(self.root, 'GT')
        self.RAW_root = ospj(self.root, 'RAW')
        
        self.GT_folder_paths = [ospj(self.GT_root, path) for path in  os.listdir(self.GT_root)] # 001 002 ...
        self.RAW_paths = [ospj(ospj(self.RAW_root, 'npydata'), path) for path in os.listdir(ospj(self.RAW_root, 'npydata'))]
        
        if self.config.task == 'olf-seg-only':
            self.GT_paths = [ospj(folder_path, 'Label_seg_olf.npy') for folder_path in self.GT_folder_paths]    
            
        # Data slicing and packaging
        #这里读取了所有的数据 并把每张切片和对应的GT进行了打包 统一存储在self.DataPackage里面
        self.DataPackage = []
        for i in range(len(self.RAW_paths)):
            GT = np.load(self.GT_paths[i])
            DCM = np.load(self.RAW_paths[i])
            
            for n in range(*OLF_SLICE[i+1]):
                self.DataPackage.append([DCM[n, :, :], GT[n, :, :]])
                # print('breakpoint')

        print("Dataset init finished")
        
        
    def __getitem__(self, index):
        p_transform = random.random()
        
        
        image, label = self.DataPackage[index]
        '''
        if p_transform > 0.3:
            image = image * 0.6 + 0.2 # 0.6 is the contrast_factor, 0.2 is the brighness_factor
        '''    
            
        image, label = Image.fromarray(image), Image.fromarray(label)
            
        aspect_ratio = image.size[1] / image.size[0]
        
        Transform = []
    
        """Resize"""
        ResizeRange = random.randint(400, max(image.size))
        Transform.append(
            T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
        
        """Random Data Augmentation"""
        
        if self.mode == 'train' and p_transform >= self.aug_prob:
            """Rotation"""
            RotationDegree = self.RotationDegree[random.randint(0, 3)]
            if RotationDegree == 90 or RotationDegree == 270:
                aspect_ratio = 1 / aspect_ratio
            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))

            Transform = T.Compose(Transform)
            image, label = Transform(image), Transform(label)
            Transform = []
            
            """Crop"""
            CropRange = random.randint(440, 480)
            if CropRange < image.size[0] and int(CropRange*aspect_ratio) < image.size[1]:
                Transform.append(T.RandomCrop((int(CropRange*aspect_ratio), CropRange), padding=0))
            
            Transform = T.Compose(Transform)
            image, label = Transform(image), Transform(label)
            
            """Shift"""
            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            label = label.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            
            """Color Jetter"""
            # Color Jetter cannot apply on one channel image
            '''Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02) # hue 色调
            image = Transform(image)'''
            
            Transform = []
        
        """Resize"""
        Transform.append(T.Resize((self.image_size, self.image_size)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
            
        image, label = Transform(image), Transform(label)
        
        """Normalize"""
        '''if self.config.normalize == True:
            Norm = T.Normalize(mean=self.mean, std=self.std)
            image = Norm(image)'''
        
        return image, label
        
        
    def __len__(self):
        return len(self.DataPackage)
    
    @staticmethod
    def npy2list(arr):
        l = []
        for n in range(arr.shape[0]):
            l.append(arr[n, :, :])
        return l
    


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
    
    