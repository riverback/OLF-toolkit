"""
因为第一个阶段我们只训练一个能分割骨化区域的模型,所以那些胸节段的标注是无意义的,
所以又写了这部分代码生成0-1的二元掩码,作为骨化区域的标签.
"""


import cv2
import os
import nibabel as nib
import numpy as np
from os.path import join as ospj

# Red -> RGB(230, 25, 72)


def nii2seg_olf_label(config):
    volume_path = config.volume_path
    output_folder = config.output_folder
    
    volume = nib.load(volume_path)
    volume_data = volume.get_fdata()
    if len(volume_data.shape) != 3:
        raise NotImplementedError("label should be a 3-D matrix")
    
    volume_data = np.flip(volume_data, 2)
    
    volume_data[volume_data > 0.] = 1
    
    l, w, h = volume_data.shape
    
    GT = []
    '''
    # 这部分代码是针对断状面重建矢状面的代码
    for x in range(l):
        gt = np.zeros([h, w])
        img = volume_data[x, :, :]
        for i in range(h):
            for j in range(w):
                gt[i, j] = img[j, i]
        GT.append(gt)            
    '''
    # 这部分代码是针对原始数据就是矢状面的情况
    for z in range(h):
        gt = np.zeros([w, l])
        img = volume_data[:, :, z]

        for i in range(l):
            for j in range(w):
                gt[j, i] = img[i, j]
        GT.append(gt)
    
    return GT


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--volume_path', type=str, default='Data/GT/003/Label.nii.gz',
                        help='xxxlabel.nii.gz is required')
    parser.add_argument('--output_folder', type=str,
                        default='Data_Seg_OLF/003/vis_GT', help='the folder to save png label')

    config = parser.parse_args()

    print(config)

    GT = np.array(nii2seg_olf_label(config))

    print(GT.shape) #(512, 601, 512) 第一个512是512份数据 

    output_path = ospj(config.output_folder, 'Label_seg_olf')
    
    np.save(output_path, GT)



    ### 查看含有骨化块的切边编号并记录 用于训练分割模型
    ### 用这部分代码的时候记得把上面的 volume_data[volume_data > 0.] = xxx 的xxx改成255
    '''GT = nii2seg_olf_label(config)'''

    '''for i in range(len(GT)):
        cv2.imwrite(ospj(config.output_folder, f'{i}.png'), GT[i]*255)'''

    print('break-point')
    
    