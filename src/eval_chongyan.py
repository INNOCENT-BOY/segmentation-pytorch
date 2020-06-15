import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from models.net import EncoderDecoderNet, SPPNet
from utils.preprocess import minmax_normalize, meanstd_normalize

# matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=r'D:\working_directory\lumber\models\chongyan')
parser.add_argument('--input_path', default=r'D:\dataset\lumber\201911_dataset\chongyan\all\voc')
parser.add_argument('--target_cls', default=3)
parser.add_argument('--confidence_threshold', default=0.5)
parser.add_argument('--area_threshold', 0)
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()
model_path = Path(args.model_path)
input_path = args.input_path
target_cls = int(args.target_cls)
confidence_threshold = float(args.confidence_threshold)
area_threshold = int(args.area_threshold)
tta_flag = args.tta

if model_path.is_file():
    model_file = model_path
    config = yaml.load(open(str(next(model_path.parent.glob('*.yaml')))))
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
    preds_np = preds_np[:, target_cls, :, :]
    preds_np[np.where(preds_np >= confidence_threshold)] = 255
    preds_np[np.where(preds_np < 255)] = 0
    return preds_np.astype(np.uint8)


def get_ind_masks(mask, fill_with=255, area_threshold=None):
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # _mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    *_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 0 if not area_threshold else area_threshold
    areas = []
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
        if area >= area_threshold:
            valid_contours.append(contour)
    ind_masks = np.zeros((len(valid_contours), *mask.shape[:2]), dtype=np.uint8)
    for i in range(len(valid_contours)):
        cv2.fillPoly(ind_masks[i], [valid_contours[i]], fill_with)
    return ind_masks, areas


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
times = []
with open(str(Path(input_path) / 'ImageSets' / 'Segmentation' / 'val.txt')) as f:
    filenames = f.readlines()
    target_files = list(map(lambda x: Path(input_path) / 'JPEGImages' / (x.strip() + '.jpg'), filenames))
areas = []
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
        t = time.time()
        pred_np = predict(img)
        times.append(time.time() - t)
    pred_np = np.array(Image.fromarray(pred_np[0]).resize((width, height)))

    # pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel, iterations=1)
    # pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_CLOSE, kernel, iterations=5)
    pred_masks, _ = get_ind_masks(pred_np, area_threshold=0)
    gt = np.array(Image.open(str(Path(input_path) / 'SegmentationClass' / img_file.with_suffix('.png').name)))
    # gt = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, kernel, iterations=10)
    # gt = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, kernel, iterations=1)
    gt_masks, _ = get_ind_masks(gt, area_threshold=0)
    areas.extend(_)
    show = False
    if show:
        max_num = max(len(pred_masks), len(gt_masks))
        fig, axes = plt.subplots(max_num + 1, 2, figsize=(12, 6 * (max_num + 1)), squeeze=False)
        axes[0][0].imshow(pred_np)
        axes[0][1].imshow(gt)
        for i in range(max_num):
            if len(pred_masks) > i:
                axes[i + 1][0].imshow(pred_masks[i])
            else:
                axes[i + 1][0].imshow(np.zeros(pred_np.shape))
            if len(gt_masks) > i:
                axes[i + 1][1].imshow(gt_masks[i])
            else:
                axes[i + 1][1].imshow(np.zeros(gt.shape))
        plt.show()
    positive, true_positive, true, recall = evaluate(pred_masks, gt_masks, 0.5)
    positives.append(positive)
    true_positives.append(true_positive)
    trues.append(true)
    recalls.append(recall)
    print(positive, true_positive, true, recall)
print(np.sum(true_positives) / np.sum(positives), np.sum(recalls) / np.sum(trues))
print(np.mean(times))
plt.hist(areas, bins=200, range=(0, 2000))
plt.show()
