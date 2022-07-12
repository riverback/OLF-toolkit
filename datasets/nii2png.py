import cv2
import os
import nibabel as nib
import numpy as np


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

# 由于是采用cv2进行图像的存储 因此保存图片时使用的颜色为BGR顺序的
BGR = {
    0.0: (0, 0, 0),
    2.0: (72, 25, 230),
    4.0: (45, 130, 245),
    6.0: (25, 255, 255),
    8.0: (75, 180, 60),
    10.0: (244, 212, 66),
    12.0: (216, 99, 67),
    14.0: (230, 50, 240),
    16.0: (212, 190, 250),
    18.0: (200, 250, 255),
    20.0: (195, 255, 170),
    22.0: (255, 190, 220),
    24.0: (117, 0, 0)
}


def nii2png(config):
    # 将标签按照阅片习惯的方向 在矢状面进行可视化
    volume_path = config.volume_path
    output_folder = config.output_folder

    volume = nib.load(volume_path)
    volume_data = volume.get_fdata()
    if len(volume_data.shape) != 3:
        raise NotImplementedError("label should be a 3-D matrix")

    volume_data = np.flip(volume_data, 2)
    l, w, h = volume_data.shape
    print(l, w, h)
    for z in range(h):
        GT_path = os.path.join(output_folder, f'gt_{z}.png')
        img = volume_data[:, :, z]
        GT = np.zeros([w, l, 3])
        for i in range(l):
            for j in range(w):
                GT[j, i, :] = BGR[img[i, j]]

        cv2.imwrite(GT_path, GT)
    '''
    # 现有数据的构建方式均为: 原始数据为xy平面 重建后在yz平面进行了标注
    for x in range(l):
        GT_path = os.path.join(output_folder, f'{x}.png')
        img = volume_data[x, :, :]

        GT = np.zeros([h, w, 3])
        for i in range(h):
            for j in range(w):
                GT[i, j, :] = BGR[img[j, i]]

        cv2.imwrite(GT_path, GT)
    '''
    print("convert is finished")


__all__ = [BGR]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--volume_path', type=str, default='Data/GT/003/IMG00012_dcm_Label.nii.gz',
                        help='xxxlabel.nii.gz is required')
    parser.add_argument('--output_folder', type=str,
                        default='vis', help='the folder to save png label')

    config = parser.parse_args()

    print(config)

    nii2png(config)
