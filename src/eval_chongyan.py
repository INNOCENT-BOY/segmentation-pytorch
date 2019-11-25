import argparse
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml
from PIL import Image

matplotlib.use('Agg')

import torch

from models.net import EncoderDecoderNet, SPPNet
from utils.preprocess import minmax_normalize, meanstd_normalize

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=r'd:\working_directory\lumber\models\chongyan')
parser.add_argument('--input_path', default=r'D:\working_directory\lumber\data\chongyan_inspect')
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()
model_path = Path(args.model_path)
input_path = args.input_path
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


def get_ind_masks(mask, fill_with=255, show=False):
    *_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ind_masks = np.ndarray((len(contours), *mask.shape[:2]), dtype=np.int)
    for i in range(len(contours)):
        cv2.fillConvexPoly(ind_masks[i], contours[i], fill_with)
    if show:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(len(contours), 1, figsize=(12, 6 * len(contours)), squeeze=False)
        plt.tight_layout()
        for ax, img in zip(axes, ind_masks):
            ax[0].imshow(img)
        plt.show()
    return ind_masks


def compute_iou(a, b):
    ious = np.zeros((a.shape[0], b.shape[0]), dtype=np.float)
    for i in range(len(a)):
        intersection = np.sum(np.bitwise_and(a[i] > 0, b > 0), axis=(1, 2))
        union = np.sum(np.bitwise_or(a[i] > 0, b > 0), axis=(1, 2))
        ious[i, :] = intersection / union
    return ious


def evaluate(pred_masks, gt_masks, iou_threshold=0.5):
    positive = len(pred_masks)
    true = len(gt_masks)
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        true_positive = 0
        recall = 0
    else:
        ious = compute_iou(pred_masks, gt_masks)
        true_positive = np.sum(np.max(ious, axis=1) >= iou_threshold)
        recall = np.sum(np.max(ious, axis=0) >= iou_threshold)
    return positive, true_positive, true, recall

positives, true_positives, trues, recalls = [], [], [], []
with open(str(Path(input_path) / 'ImageSets' / 'Segmentation' / 'val.txt')) as f:
    filenames = f.readlines()
    target_files = list(map(lambda x: Path(input_path) / 'JPEGImages' / (x.strip() + '.jpg'), filenames))
for img_file in target_files:
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), f'处理图片{img_file.name}')
    img = Image.open(img_file)
    width, height = img.size
    img = img.resize(target_size[::-1])
    img = np.array(img)
    if net_type == 'unet':
        img = minmax_normalize(img)
        img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        img = minmax_normalize(img, norm_range=(-1, 1))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.FloatTensor(img)
    with torch.no_grad():
        pred_np = predict(img)
    pred_np = np.array(Image.fromarray(pred_np[0]).resize((width, height)))
    pred_masks = get_ind_masks(pred_np)
    gt_masks = get_ind_masks(np.array(Image.open(str(Path(input_path) / 'SegmentationClass' / img_file.with_suffix('.png').name))))
    positive, true_positive, true, recall = evaluate(pred_masks, gt_masks, 0.5)
    positives.append(positive)
    true_positives.append(true_positive)
    trues.append(true)
    recalls.append(recall)
    print(positive, true_positive, true, recall)
print(np.sum(true_positives) / np.sum(positives), np.sum(recalls) / np.sum(trues))