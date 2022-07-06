import os
from os.path import join as ospj
import numpy as np
import SimpleITK as sitk
import pydicom
import cv2


def ReadImageSequenceSitk(data_folder):
    data_paths = os.listdir(data_folder)

    itk_img_head = sitk.ReadImage(ospj(data_folder, data_paths[0]))
    data = sitk.GetArrayFromImage(itk_img_head)

    for path in data_paths[1:]:
        itk_img = sitk.ReadImage(ospj(data_folder, path))
        img = sitk.GetArrayFromImage(itk_img)

        data = np.concatenate((data, img), axis=0)

    data = np.transpose(data, (2, 1, 0))

    slices = [np.rot90(np.flip(data[idx, :, :], axis=1), 1)
              for idx in range(data.shape[0])]

    return slices


def load_scan(dcm_folder):
    dcm_paths = os.listdir(dcm_folder)
    slices = [pydicom.read_file(ospj(dcm_folder, dcm_path))
              for dcm_path in dcm_paths]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def normalize_minmax(arr):
    min_v, max_v = arr.min(), arr.max()
    return (arr - min_v) / (max_v - min_v)


def ReadDcmSequencePydicom(dcm_folder, norm=True):
    scans = load_scan(dcm_folder)
    hu_scans = get_pixels_hu(scans)
    if norm == True:
        hu_scans = normalize_minmax(hu_scans)
        # hu_scans = 255 * hu_scans

    hu_scans = np.transpose(hu_scans, (2, 1, 0))
    slices = [np.rot90(np.flip(hu_scans[idx, :, :], axis=1), 1)
              for idx in range(hu_scans.shape[0])]

    '''
    vis_path = r'C:\ZhuangResearchCode\OLF_TASK\vis'
    for i in range(len(slices)):
        out_path = ospj(vis_path, f'{i}.png')
        cv2.imwrite(out_path, slices[i])
    print("finished")
    '''

    return slices


__all__ = [ReadImageSequenceSitk]


if __name__ == '__main__':
    data_folder = r'..\Data\RAW\002'

    arr = np.array(ReadDcmSequencePydicom(data_folder))
    np.save(ospj(r'..\Data\RAW\npydata', '002'), arr)

    print("hello-world")
