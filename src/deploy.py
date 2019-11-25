import argparse
from pathlib import Path
import time
from itertools import chain

import matplotlib
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use('Agg')

import torch
import albumentations as albu

from models.net import EncoderDecoderNet, SPPNet
from utils.preprocess import minmax_normalize, meanstd_normalize
from utils.custum_aug import PadIfNeededRightBottom


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='d:/working_directory/lumber/models/chongyan')
parser.add_argument('--input_path', default='d:/working_directory/lumber/inputs')
parser.add_argument('--output_path', default='d:/working_directory/lumber/outputs')
parser.add_argument('--target_file_patterns', default='zhengguang*.bmp')
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()
model_path = Path(args.model_path)
input_path = args.input_path
Path(input_path).mkdir(exist_ok=True, parents=True)
output_path = args.output_path
Path(output_path).mkdir(exist_ok=True, parents=True)
target_file_patterns = args.target_file_patterns.split('|')
tta_flag = args.tta

config = yaml.load(open(str(next(model_path.glob('*.yaml')))))
net_config = config['Net']
net_config['output_channels'] = config['Data']['num_classes']
target_size = config['Data']['target_size']
target_size = eval(target_size)
ignore_index = config['Loss']['ignore_index']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_file = Path(model_path) / 'model.pth'

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

model.eval()

batch_size = 1
scales = [0.25, 0.75, 1, 1.25]

def predict(images):
    images = images.to(device)
    if tta_flag:
        preds = model.tta(images, scales=scales, net_type=net_type)
    else:
        preds = model.pred_resize(images, images.shape[2:], net_type=net_type)
    preds = preds.argmax(dim=1)
    preds_np = preds.detach().cpu().numpy().astype(np.uint8)
    return preds_np


while True:
    time.sleep(0.01)
    target_files = [f for p in target_file_patterns for f in Path(input_path).glob(p)]
    for img_file in target_files:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), f'处理图片{img_file.name}')
        try:
            img = Image.open(img_file)
            sub_imgs = []
            width, height = img.size
            parts = 2
            sub_width, sub_height = width // parts, height // parts
            for i in range(parts):
                for j in range(parts):
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
                preds_np = predict(sub_imgs)
            result = np.ndarray((sub_height * parts, sub_width * parts), dtype=np.uint8)
            for i in range(parts):
                for j in range(parts):
                    sub_img = np.array(Image.fromarray(preds_np[i + j * parts]).resize((sub_width, sub_height)))
                    result[sub_height * i: sub_height * (i + 1), sub_width * j: sub_width * (j + 1)] = sub_img[:]
            # result[np.where(result==3)] = 255
            Image.fromarray(result, mode='P').save(str(Path(output_path) / img_file.with_suffix('.png').name))
            try:
                img_file.unlink()
            except FileNotFoundError:
                pass
        except OSError as e:
            print(e)

