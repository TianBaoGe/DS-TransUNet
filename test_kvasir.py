import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from lib.DS_TransUNet import UNet

from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader,test_dataset
from PIL import Image

pred_path = 'output/kvasir/pred/'
gt_path = 'output/kvasir/gt/'

def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    n_val = len(loader)
    pred_idx=0
    gt_idx=0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred, _, _ = net(imgs)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            for img in pred:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(pred_path+'/'+str(pred_idx)+'.png')
                pred_idx += 1
            for img in true_masks:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(gt_path+'/'+str(gt_idx)+'.png')
                gt_idx += 1

            pbar.update()


def test_net(net,
              device,
              batch_size=1,
              n_class=1,
              img_size=512):


    val_img_dir = 'data/Kvasir_SEG/val/images/'
    val_mask_dir = 'data/Kvasir_SEG/val/masks/'

    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    net.eval()

    eval_net(net, val_loader, device)


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default='checkpoints/kvasir.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=512,
                        help='The size of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(128, 1)
    net = nn.DataParallel(net, device_ids=[0])
    net.to(device=device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device), False
        )
        logging.info(f'Model loaded from {args.load}')

    try:
        test_net(net=net,
                  batch_size=args.batchsize,
                  device=device,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
