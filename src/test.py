import argparse
import time
from pathlib import Path
import os

import glob
import matplotlib
import numpy as np
import yaml
from yaml import Loader
from PIL import Image
import cv2

matplotlib.use('Agg')

import torch

from models.net import EncoderDecoderNet, SPPNet
from utils.preprocess import minmax_normalize, meanstd_normalize

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=r'd:\working_directory\lumber\models\qiaopi\model_0040.pth')
parser.add_argument('--input_path', default='/home/lijiahui/data/duanlie/TestImages')
parser.add_argument('--output_path', default='/home/lijiahui/data/duanlie/MaskImages')
parser.add_argument('--target_file_patterns', default='*.bmp')
parser.add_argument('--target_cls', default=2)
parser.add_argument('--width_parts', default=1)
parser.add_argument('--height_parts', default=1)
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()
model_path = Path(args.model_path)
input_path = args.input_path
output_path = args.output_path
Path(output_path).mkdir(exist_ok=True, parents=True)
target_file_patterns = args.target_file_patterns.split('|')
target_cls = int(args.target_cls)
width_parts = int(args.width_parts)
height_parts = int(args.height_parts)
tta_flag = args.tta

if model_path.is_file():
    model_file = model_path
    config = yaml.load(open(str(next(model_path.parent.glob('*.yaml')))), Loader=Loader)
else:
    model_file = sorted(list(model_path.glob('model*_best.pth')))[-1]
    config = yaml.load(open(str(next(model_path.glob('*.yaml')))))
net_config = config['Net']
net_config['output_channels'] = config['Data']['num_classes']
target_size = config['Data']['target_size']
target_size = eval(target_size)
ignore_index = config['Loss']['ignore_index']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if 'unet' in net_config['dec_type']:
    net_type = 'unet'
    model = EncoderDecoderNet(**net_config)
else:
    net_type = 'deeplab'
    model = SPPNet(**net_config)
model.to(device)
if 'unet' not in net_config['dec_type']:
    model.update_bn_eps()

param = torch.load(model_file)
model.load_state_dict(param)
del param
print("model loaded successful!")

model.eval()

batch_size = 1
scales = [0.25, 0.75, 1, 1.25]


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def predict(images):
    images = images.to(device)
    if tta_flag:
        preds = model.tta(images, scales=scales, net_type=net_type)
    else:
        preds = model.pred_resize(images, images.shape[2:], net_type=net_type)
    # preds = preds.argmax(dim=1)
    preds_np = preds.detach().cpu().numpy()
    preds_np = softmax(preds_np, axis=1)

    # print(preds_np.shape)
    return preds_np

# target_files = [f for p in target_file_patterns for f in Path(input_path).glob(p)]
# target_files = glob.glob(input_path+'/*png')
target_files = [os.path.join(input_path, im) for im in os.listdir(input_path)]
print(target_files)
for img_file in target_files:
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), f'处理图片{img_file.name}')
    img = Image.open(img_file)
    sub_imgs = []
    width, height = img.size
    sub_width, sub_height = width // width_parts, height // height_parts
    for i in range(width_parts):
        for j in range(height_parts):
            sub_img = img.crop((sub_width * i, sub_height * j, sub_width * (i + 1), sub_height * (j + 1)))
            sub_img = sub_img.resize(target_size[::-1])
            sub_img = np.array(sub_img)
            if net_type == 'unet':
                sub_img = minmax_normalize(sub_img)
                sub_img = meanstd_normalize(sub_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                sub_img = minmax_normalize(sub_img, norm_range=(-1, 1))
            sub_img = np.transpose(sub_img, (2, 0, 1))
            sub_imgs.append(sub_img)
    sub_imgs = np.stack(sub_imgs)
    sub_imgs = torch.FloatTensor(sub_imgs)
    with torch.no_grad():
        preds_np = predict(sub_imgs)[:, target_cls, :, :]
    mask = np.ndarray((sub_height * height_parts, sub_width * width_parts), dtype=np.uint8)
    for i in range(width_parts):
        for j in range(height_parts):
            sub_mask = cv2.resize(preds_np[i * height_parts + j], (sub_width, sub_height))
            mask[sub_height * j: sub_height * (j + 1), sub_width * i: sub_width * (i + 1)] = np.round(sub_mask * 255)
    # filename = '.'.join(img_file.name.split('.')[:-1])
    filename = '.'.join(img_file.split('/')[-1].split('.')[:-1])
    img.crop((0, 0, sub_width * width_parts, sub_height * height_parts)).save(str(Path(output_path) / (filename + '_part1.png')))
    Image.fromarray(mask).save(str(Path(output_path) / (filename + '_part2.png')))
