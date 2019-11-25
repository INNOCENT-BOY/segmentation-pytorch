import cv2
import numpy as np
from PIL import Image

file1 = r'D:\dataset\lumber\201911_dataset\ceguang_qiaopi\voc\SegmentationClass\qiaopi_2019-11-12_10.30.20.008714_zuoguang_10.png'
file2 = r'D:\dataset\lumber\201911_dataset\ceguang_qiaopi\voc\SegmentationClass\qiaopi_2019-11-12_10.31.10.229681_zuoguang_00.png'

gt_mask = np.array(Image.open(file1))
mask = np.array(Image.open(file2))


def get_ind_masks(mask, fill_with=255, show=False):
    *_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ind_masks = np.ndarray((len(contours), *gt_mask.shape[:2]), dtype=np.int)
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


gt_ind_masks = get_ind_masks(gt_mask)
ind_masks = get_ind_masks(mask)


def compute_iou(a, b):
    ious = np.zeros((a.shape[0], b.shape[0]), dtype=np.float)
    for i in range(len(a)):
        intersection = np.sum(np.bitwise_and(a[i] > 0, b > 0), axis=(1, 2))
        union = np.sum(np.bitwise_or(a[i] > 0, b > 0), axis=(1, 2))
        ious[i, :] = intersection / union
    return ious


def evaluate(pred_masks, gt_masks, iou_threshold=0.5):
    ious = compute_iou(pred_masks, gt_masks)
    accuracy = np.sum(np.max(ious, axis=1) >= iou_threshold) / len(pred_masks)
    recall = np.sum(np.max(ious, axis=0) >= iou_threshold) / len(gt_masks)
    return accuracy, recall

prs = [evaluate(ind_masks, gt_ind_masks, iou) for iou in np.arange(0, 1, 0.1)]
print(prs)