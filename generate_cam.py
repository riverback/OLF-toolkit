"""
# Define your model
from torchvision.models import resnet18
model = resnet18(pretrained=True).eval()

# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)
"""

"""
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

model = resnet18(pretrained=True).eval()
cam_extractor = SmoothGradCAMpp(model)
# Get your input
img = read_image("path/to/your/image.png")
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
"""

"""
# If you want to visualize your heatmap, 
# you only need to cast the CAM to a numpy ndarray:
import matplotlib.pyplot as plt
# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
"""
import argparse
import sys
import os
from os.path import join as ospj
from PIL import Image
import torch 
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from model.build_model import build_model
from datasets.prepare_classify_and_cam import ReadDcmSequence_XY_Pydicom
from utils.Logger import Logger

from torchcam.methods import ScoreCAM, SmoothGradCAMpp, CAM, LayerCAM, XGradCAM, ISCAM, GradCAM, GradCAMpp, SSCAM
from torchcam.utils import overlay_mask

label2class = [
    'Healthy',
    'OLF',
    'DO'
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--image_channels', type=int, default=1)
    parser.add_argument('--output_channels', type=int, default=3)
    parser.add_argument('--cuda_idx', type=str, default='7')

    parser.add_argument('--checkpoint_path', type=str, default='experiments/ResNet50-CosineAnnealingLR-epoch80/checkpoints/checkpoint_70.pth')

    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_idx
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.checkpoint_path)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    cam_extractor = GradCAM(model=model, target_layer='layer4')

    vis_folder = ospj('vis', 'cam', '002')
    wrong_folder = ospj(vis_folder, 'wrong')
    right_folder = ospj(vis_folder, 'right')
    
    if not os.path.exists(wrong_folder):
        os.makedirs(wrong_folder)

    if not os.path.exists(right_folder):
        os.makedirs(right_folder)
    

    Data_root = 'Data'

    data_index = '002'

    label_path = ospj(Data_root, 'GT', data_index, 'Label_class3_xy.txt')
    image_path = ospj(Data_root, 'RAW', data_index)

    images = ReadDcmSequence_XY_Pydicom(image_path) # ndarray [N, W, H]    
    targets = []
    with open(label_path, 'r') as f:
        label_data = f.readlines()
        for index in range(len(label_data)):
            p = ospj(image_path, 'IMG{:05d}.dcm'.format(index))
            label = int(label_data[index].strip('\n').split()[-1])
            targets.append([label, p])

    predisright = True

    for index in range(images.shape[0]):

        image = images[index]
        image_tensor = torch.tensor(image).type(torch.FloatTensor).to(device)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        label, img_path = targets[index]
        label = torch.tensor(label).long().to(device)

        '''
        plt.imshow(image)
        plt.savefig('vis/test.png', cmap='gray')
        '''

        out = model(image_tensor)
        out = torch.softmax(out, dim=1)

        # print("out label: {}, ground truth: {}".format(out.squeeze(0).argmax().item(), label.item()))

        if out.squeeze(0).argmax() == label:
            predisright = True
        else:
            predisright = False

        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        # print(activation_map[0].max())

        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title('input')
        plt.tight_layout()
        plt.show()

        # print(type(activation_map))
        
        
        image = np.expand_dims(image, 2)
        image = np.concatenate((image, image, image), axis=2).astype(np.uint8)
        # print(image.shape)

        # image = Image.fromarray((image*255).astype('uint8'))
        cam_results = overlay_mask(to_pil_image(image*255), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
        # print(type(cam_results))

        # print(cam_results.size)

        
        plt.subplot(132)
        plt.imshow(activation_map[0].cpu().numpy())
        plt.axis('off')
        plt.title('cam layer 4')
        plt.tight_layout()
        plt.show()
        
        plt.subplot(133)
        plt.imshow(cam_results)
        plt.axis('off')
        plt.title('CAM')
        plt.tight_layout()
        plt.show()

        plt.suptitle('pred: {}: {:.2f}, {}: {:.2f}, {}: {:.2f}, label: {}'
            .format(label2class[0], out.cpu().squeeze(0)[0], label2class[1], out.cpu().squeeze(0)[1], label2class[2], out.cpu().squeeze(0)[2], label2class[label.cpu()]))

        if predisright:
            plt.savefig(ospj(right_folder, 'IMG{:05d}.png'.format(index)))
        else:
            plt.savefig(ospj(wrong_folder, 'IMG{:05d}.png'.format(index)))

        # print("breakpoint")

