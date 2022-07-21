"""
生成用于横断面分类+CAM机器视觉的数据
"""

import numpy as np
import cv2
import nibabel as nib
from os.path import join as ospj

from datasets.dcmutils import load_scan, normalize_minmax, get_pixels_hu


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
    



if __name__ == '__main__':

    DO_Spine = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    label_path = 'Data/GT/001/Label.nii.gz'
    dcm_folder = 'Data/RAW/001'
    label_classify_xy_path = '001.txt'
    generate_class3_label(dcm_folder, label_path, label_classify_xy_path, DO_Spine)


    DO_Spine = [18.0]
    label_path = 'Data/GT/002/Label.nii.gz'
    dcm_folder = 'Data/RAW/002'
    label_classify_xy_path = '002.txt'
    generate_class3_label(dcm_folder, label_path, label_classify_xy_path, DO_Spine)

    

