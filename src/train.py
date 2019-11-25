import argparse
import pickle
from collections import OrderedDict
from pathlib import Path
import shutil
import math

import albumentations as albu
import numpy as np
import torch
import yaml
from logger.log import debug_logger
from logger.plot import history_ploter
from losses.multi import MultiClassCriterion
from matplotlib import pyplot as plt
from models.net import EncoderDecoderNet, SPPNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import compute_iou_batch
from utils.optimizer import create_optimizer
from utils.preprocess import minmax_normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))
    net_config = config['Net']
    data_config = config['Data']
    num_classes = data_config['num_classes']
    net_config['output_channels'] = data_config['num_classes']
    train_config = config['Train']
    loss_config = config['Loss']
    opt_config = config['Optimizer']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t_max = opt_config['t_max']

    max_epoch = train_config['max_epoch']
    batch_size = train_config['batch_size']
    fp16 = train_config['fp16']
    resume = train_config['resume']
    pretrained_path = train_config['pretrained_path']
    output_dir = Path(train_config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(str(config_path), str(output_dir / config_path.name))

    log_dir = Path(train_config['log_dir'])
    log_dir.mkdir(exist_ok=True, parents=True)
    eval_every_n_epochs = train_config['eval_every_n_epochs']
    vis_flag = train_config['vis_flag']

    include_bg = loss_config['include_bg']
    del loss_config['include_bg']

    # Network
    if 'unet' in net_config['dec_type']:
        net_type = 'unet'
        model = EncoderDecoderNet(**net_config)
    else:
        net_type = 'deeplab'
        model = SPPNet(**net_config)

    dataset = data_config['dataset']
    if dataset == 'pascal':
        from dataset.pascal_voc import PascalVocDataset as Dataset
    elif dataset == 'cityscapes':
        from dataset.cityscapes import CityscapesDataset as Dataset
    else:
        raise NotImplementedError
    if include_bg:
        classes = np.arange(0, data_config['num_classes'])
    else:
        classes = np.arange(1, data_config['num_classes'])
    del data_config['dataset']
    del data_config['num_classes']

    logger = debug_logger(log_dir)
    logger.debug(config)
    logger.info(f'Device: {device}')
    logger.info(f'Max Epoch: {max_epoch}')

    # Loss
    loss_fn = MultiClassCriterion(**loss_config).to(device)
    params = model.parameters()
    optimizer, scheduler = create_optimizer(params, **opt_config)

    # history
    if resume:
        with open(log_dir.joinpath('history.pkl'), 'rb') as f:
            history_dict = pickle.load(f)
            best_metrics = history_dict['best_metrics']
            loss_history = history_dict['loss']
            iou_history = history_dict['iou']
            start_epoch = len(iou_history)
            for _ in range(start_epoch):
                scheduler.step()
    else:
        start_epoch = 0
        best_metrics = 0
        loss_history = []
        iou_history = []

    # Dataset
    affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                     # Rotate(5, p=.5)
                                     ])
    # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
    #                                 albu.RandomBrightnessContrast(p=.5)])
    image_augmenter = None
    train_dataset = Dataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                            split='train', net_type=net_type, **data_config)
    valid_dataset = Dataset(split='valid', net_type=net_type, **data_config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # To device
    model = model.to(device)

    # Pretrained model
    if pretrained_path:
        logger.info(f'Resume from {pretrained_path}')
        param = torch.load(pretrained_path)
        model.load_state_dict(param)
        del param

    # fp16
    if fp16:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        logger.info('Apply fp16')

    # Restore model
    if resume:
        model_path = sorted(list(output_dir.glob('model_*.pth')))[-1]
        # model_path = output_dir.joinpath(f'model_tmp.pth')
        logger.info(f'Resume from {model_path}')
        param = torch.load(model_path)
        model.load_state_dict(param)
        del param
        opt_path = sorted(list(output_dir.glob('opt_*.pth')))[-1]
        # opt_path = output_dir.joinpath(f'opt_tmp.pth')
        param = torch.load(opt_path)
        optimizer.load_state_dict(param)
        del param

    # Train
    for i_epoch in range(start_epoch, max_epoch):
        logger.info(f'Epoch: {i_epoch}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_losses = []
        train_ious = []
        model.train()
        with tqdm(train_loader) as _tqdm:
            for batched in _tqdm:
                images, labels, _ = batched
                if fp16:
                    images = images.half()
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = model(images)
                if net_type == 'deeplab':
                    preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
                loss = loss_fn(preds, labels)

                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes[1 if include_bg else 0:])

                _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
                train_losses.append(loss.item())
                train_ious.append(iou)

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

        scheduler.step()

        train_loss = np.mean(train_losses)
        train_iou = np.nanmean(train_ious)
        logger.info(f'train loss: {train_loss}')
        logger.info(f'train iou: {train_iou}')

        torch.save(model.state_dict(), output_dir.joinpath(f'model_{i_epoch:04d}.pth'))
        torch.save(optimizer.state_dict(), output_dir.joinpath(f'opt_{i_epoch:04d}.pth'))

        if (i_epoch + 1) % eval_every_n_epochs == 0:
            valid_losses = []
            valid_ious = []
            images_list = []
            labels_list = []
            preds_list = []
            fp_list = []
            model.eval()
            with torch.no_grad():
                with tqdm(valid_loader) as _tqdm:
                    for batched in _tqdm:
                        images, labels, _ = batched
                        if fp16:
                            images = images.half()
                        images, labels = images.to(device), labels.to(device)
                        preds = model.tta(images, net_type=net_type)
                        if fp16:
                            loss = loss_fn(preds.float(), labels)
                        else:
                            loss = loss_fn(preds, labels)

                        images_np = images.detach().cpu().numpy()
                        images_list.append(images_np)
                        preds_np = preds.detach().cpu().numpy()
                        preds_list.append(preds_np)
                        labels_np = labels.detach().cpu().numpy()
                        labels_list.append(labels_np)
                        iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes[1 if include_bg else 0:])
                        fp = np.sum(np.bitwise_and(np.argmax(preds_np, axis=1) != 0, labels_np == 0))
                        fp_list.append(fp)

                        _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}', fp=f'{fp}'))
                        valid_losses.append(loss.item())
                        valid_ious.append(iou)

            valid_loss = np.mean(valid_losses)
            valid_iou = np.nanmean(valid_ious)
            valid_fp = np.mean(fp)
            logger.info(f'valid seg loss: {valid_loss}')
            logger.info(f'valid iou: {valid_iou}')
            logger.info(f'valid false positive: {valid_fp}')

            if best_metrics < valid_iou:
                best_metrics = valid_iou
                logger.info('Best Model!')
                # torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
                # torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))

            if vis_flag:
                images = np.concatenate(images_list)
                images = np.transpose(images, (0, 2, 3, 1))
                labels = np.concatenate(labels_list)
                preds = np.concatenate(preds_list)
                preds = np.argmax(preds, axis=1)

                ignore_pixel = labels == loss_config['ignore_index']
                preds[ignore_pixel] = num_classes
                labels[ignore_pixel] = num_classes

                num_per_visualization = 100
                for i in range(math.ceil(len(images) / float(num_per_visualization))):
                    start = i * num_per_visualization
                    num = min(num_per_visualization, len(images) - start)
                    fig, axes = plt.subplots(num, 3, figsize=(12, 3 * num))
                    plt.tight_layout()

                    axes[0, 0].set_title('input image')
                    axes[0, 1].set_title('prediction')
                    axes[0, 2].set_title('ground truth')

                    for ax, img, lbl, pred in zip(axes, images[start: start + num], labels[start: start + num], preds[start: start + num]):
                        if net_type == 'unet':
                            mean = np.asarray([0.485, 0.456, 0.406])
                            std = np.asarray([0.229, 0.224, 0.225])
                            img = img * std + mean
                            img = np.clip(img, 0., 1.)
                        else:
                            img = minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1))
                        ax[0].imshow(img)
                        ax[1].imshow(pred)
                        ax[2].imshow(lbl)
                        ax[0].set_xticks([])
                        ax[0].set_yticks([])
                        ax[1].set_xticks([])
                        ax[1].set_yticks([])
                        ax[2].set_xticks([])
                        ax[2].set_yticks([])

                    (log_dir / 'eval_vis').mkdir(exist_ok=True, parents=True)
                    plt.savefig(
                        str(
                            log_dir / 'eval_vis' / f'{i_epoch:04d}_{valid_iou:.4f}_{valid_fp}{"_best" if best_metrics == valid_iou else ""}_{i:03d}.png'))
                    plt.close()
        else:
            valid_loss = None
            valid_iou = None

        loss_history.append([train_loss, valid_loss])
        iou_history.append([train_iou, valid_iou])
        history_ploter(loss_history, log_dir.joinpath('loss.png'))
        history_ploter(iou_history, log_dir.joinpath('iou.png'))

        history_dict = {'loss': loss_history,
                        'iou': iou_history,
                        'best_metrics': best_metrics}
        with open(log_dir.joinpath('history.pkl'), 'wb') as f:
            pickle.dump(history_dict, f)
