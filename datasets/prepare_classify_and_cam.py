"""
生成用于横断面分类+CAM机器视觉的数据
"""

import os
import numpy as np
import cv2
import nibabel as nib
from os.path import join as ospj
from dcmutils import load_scan, normalize_minmax, get_pixels_hu
# from datasets.dcmutils import load_scan, normalize_minmax, get_pixels_hu


def ReadDcmSequence_XY_Pydicom(dcm_folder, norm=True):
    # 这样得到的数据是从头部往下的标签
    # dim=0 对应z轴
    scans = load_scan(dcm_folder)
    hu_scans = get_pixels_hu(scans)
    if norm:
        hu_scans = normalize_minmax(hu_scans)
        # 如果需要可视化 要再乘255
        # hu_scans *= 255
    '''
    vis_path = 'vis'
    out_path = ospj(vis_path, f'raw.png')
    cv2.imwrite(out_path, hu_scans[0, :, :])
    '''
    return hu_scans


def ReadNiiLabel(label_path):
    volume_path = label_path
    volume = nib.load(volume_path)
    volume_data = volume.get_fdata()

    # dim=2 对应z轴
    # 默认的标签 z轴索引 索引大的靠近头部
    volume_data = np.flip(volume_data, 2) # 翻转z轴 
    volume_data = np.transpose(volume_data, (2, 0, 1))
    
    return volume_data

def generate_class3_label(dcm_folder, label_path, label_classify_xy_path='label_classify_xy.txt', DO_Label_Value=[]):
    dcms = ReadDcmSequence_XY_Pydicom(dcm_folder)
    labels = ReadNiiLabel(label_path)
    output_path = ospj('vis', label_classify_xy_path)

    if labels.shape[0] != dcms.shape[0]:
        raise ValueError("z axis should be the same")

    """
    0 - others
    1 - olf
    2 - do
    """
    
    with open(output_path, mode='w') as f:
        for z in range(labels.shape[0]):
            if labels[z, :, :].max() in DO_Label_Value:
                f.write(f'dcm{z} {2}\n')
            elif labels[z, :, :].max() > 0:
                f.write(f'dcm{z} {1}\n')
            else:
                f.write(f'dcm{z} {0}\n')
    
    print("finish")
    


def resize_ori_image_XY(dcm_folder, label_path, DO_Label_Value=[], folder='001'):
    dcms = ReadDcmSequence_XY_Pydicom(dcm_folder)
    labels = ReadNiiLabel(label_path)
    output_folder = 'png_resized'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder = ospj('png_resized', folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    labels[labels > 0] = 1

    w, h = 256, 256
    

    for n in range(dcms.shape[0]):
        dcm = dcms[n]
        if dcm.shape[0] != dcm.shape[1]:
            raise ValueError("dcm is not a square with same w and h")
        png_resized = np.zeros([w, h])
        save_path = os.path.join(output_folder, '{:05d}IMG.png'.format(n))
        if labels[n].max() > 0:
            position = labels[n].argmax()
            x = position // labels[n].shape[0] - 175
            y = position % labels[n].shape[0] - 100
            if x + w > labels[n].shape[0]:
                x = labels[n].shape[0] - 224
            if y + h > labels[n].shape[1]:
                y = labels[n].shape[1] - 224
        else:
            x = 100
            y = 150
        
        x = 135
        y = 125

        png_resized = dcm[x:x+w, y:y+w]

        if png_resized.max() <= 1.:
            png_resized = (normalize_minmax(png_resized) * 255).astype(np.uint8)

        cv2.imwrite(save_path, png_resized)




    


if __name__ == '__main__':

    DO_Spine = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    label_path = 'Data/GT/001/Label.nii.gz'
    dcm_folder = 'Data/RAW/001'
    label_classify_xy_path = '001.txt'
    resize_ori_image_XY(dcm_folder, label_path, DO_Spine, '001')
    # generate_class3_label(dcm_folder, label_path, label_classify_xy_path, DO_Spine)

    
    DO_Spine = [18.0]
    label_path = 'Data/GT/002/Label.nii.gz'
    dcm_folder = 'Data/RAW/002'
    label_classify_xy_path = '002.txt'
    # resize_ori_image_XY(dcm_folder, label_path, DO_Spine, '002')
    # generate_class3_label(dcm_folder, label_path, label_classify_xy_path, DO_Spine)
    
   

