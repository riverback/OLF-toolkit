"""
这一部分的代码是用来生成分割olf+do的标签的
打算做成下面的形式, 
l, w, h = volume_data.shape
label = [h, w]
label = 1 -> olf
label = 2 -> do
"""


"""标签颜色记录表 一共12种颜色 对应胸椎十二个节段
FROM: https://sashamaps.net/docs/resources/20-colors/

Color       Hexadecimal code     Thethoracic spine     Label Value           RGB
Red         #e61948                     T1              2.0             (230, 25, 72)
Orange      #f58231                     T2              4.0             (245, 130, 49)
Yellow      #ffe119                     T3              6.0             (255, 255, 25)
Green       #3cb44b                     T4              8.0             (60, 180, 75)
Cyan        #42d4f4                     T5              10.0            (66, 212, 244)
Blue        #4363d8                     T6              12.0            (67, 99, 216)
Magenta     #f032e6                     T7              14.0            (240, 50, 230)
Pink        #fabed4                     T8              16.0            (250, 190, 212)
Beige       #fffac8                     T9              18.0            (255, 250, 200)
Mint        #aaffc3                     T10             20.0            (170, 255, 195)
Lavender    #dcbeff                     T11             22.0            (220, 190, 255)
Nacy        #000075                     T12             24.0            (0, 0, 117)

"""


import os
import nibabel as nib
import numpy as np
from os.path import join as ospj
import torch

def nii2seg_olf_and_do_label(volume_path, DO_Spine, flag='xy'):
    volume = nib.load(volume_path)
    volume_data = volume.get_fdata()
    if len(volume_data.shape) != 3:
        raise NotImplementedError("label should be a 3-D matrix")

    volume_data = np.flip(volume_data, 2) # 因为z轴和习惯的方向是反过来的

    

    l, w, h = volume_data.shape

    print(volume_data.shape)

    GT = []

    ### 断状面(xy)重建矢状面的代码
    if flag == 'xy':
        for x in range(l):
            gt = np.zeros([3, h, w])
            img = volume_data[x, :, :]
            if img.max() < 0.5:
                gt[0, :, :] = 1
                GT.append(gt)
                continue
            for i in range(h):
                for j in range(w):
                    if img[j, i] in DO_Spine:
                        gt[2, i, j] = 1
                    elif img[j, i] > 0:
                        gt[1, i, j] = 1
                    else:
                        gt[0, i, j] = 1
            GT.append(gt)

    ### 
    else:
        for z in range(h):
            gt = np.zeros([3, w, l])
            img = volume_data[:, :, z]
            if img.max() < 0.5:
                gt[0, :, :] = 1
                GT.append(gt)
                continue
            for i in range(l):
                for j in range(w):
                    if img[i, j] in DO_Spine:
                        gt[2, j, i] = 1
                    elif img[i, j] > 0:
                        gt[1, j, i] = 1
                    else:
                        gt[0, j, i] = 1
            GT.append(gt)
    

    return GT


if __name__ == '__main__':

    volume_path = 'Data/GT/002/Label.nii.gz'
    output_folder = 'vis'
    DO_Spine = [18.0] 
    flag='xy'

    GT = np.array(nii2seg_olf_and_do_label(volume_path, DO_Spine, flag))

    print(GT.shape)

    # save_path = os.path.join(output_folder, 'Label_seg_olf_and_do_2')
    save_path = 'Data/GT/002/Label_seg_olf_and_do.npy'


    np.save(save_path, GT)

    print("finish")

##########################
    
    volume_path = 'Data/GT/001/Label.nii.gz'
    output_folder = 'vis'
    DO_Spine = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0] 
    flag='xy'

    GT = np.array(nii2seg_olf_and_do_label(volume_path, DO_Spine, flag))

    print(GT.shape)

    # save_path = os.path.join(output_folder, 'Label_seg_olf_and_do_1')
    save_path ='Data/GT/001/Label_seg_olf_and_do.npy'

    np.save(save_path, GT)

    print("finish")
