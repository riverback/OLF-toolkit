import os
from os.path import join as ospj
import numpy as np
import SimpleITK as sitk


def ReadImageSequence(data_folder):
    data_paths = os.listdir(data_folder)

    itk_img_head = sitk.ReadImage(ospj(data_folder, data_paths[0]))
    data = sitk.GetArrayFromImage(itk_img_head)

    for path in data_paths[1:]:
        itk_img = sitk.ReadImage(ospj(data_folder, path))
        img = sitk.GetArrayFromImage(itk_img)

        data = np.concatenate((data, img), axis=0)

    data = np.transpose(data, (2, 1, 0))

    return data


__all__ = [ReadImageSequence]


if __name__ == '__main__':
    data_folder = ''
    data = ReadImageSequence(data_folder)
    idx = 42
    slice_idx = data[idx, :, :]
    slice_idx = np.flip(slice_idx, axis=1)  # horizontal flip
    slice_idx = np.rot90(slice_idx, 1)  # anti-clock 90°

    # now: slice <--> label
