import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser, ArgumentTypeError

import torch
from torch import cuda
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from dataset import SceneTextDataset_val
from model import EAST

######################
import wandb
from datetime import datetime
######################

######################
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')
#######################


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    ########################################
    parser.add_argument('--validation', type=str2bool, default=True)
    parser.add_argument('--train_dir', type=str, default="train")
    parser.add_argument('--val_dir', type=str, default="annotation_0")
    parser.add_argument('--load_state', type=str, help="Select .pth Weight File name in model_dir.", default="")
    parser.add_argument('--exp_name', type=str, default=f'exp_{datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")}')
    ########################################

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, validation, train_dir, val_dir, load_state, exp_name):
    dataset = SceneTextDataset(data_dir, split=train_dir, image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    ################################################################################
    if len(load_state):
        model.load_state_dict(torch.load(osp.join(model_dir, f"{load_state}.pth")))
    ################################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    # best_val_acc = 0
    best_val_loss = np.inf
    # best_val_f1 = 0
    early_stop_value = 20

    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()

        # Epoch 별 Loss 체크용
        epoch_cls_loss = 0.
        epoch_angle_loss = 0.
        epoch_iou_loss = 0.

        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                # Epoch 별 Loss 체크용
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'],
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss'],
                    "Lerning Rate": optimizer.param_groups[0]["lr"]
                }
                pbar.set_postfix(val_dict)

        ######################################################
        epoch_loss /= num_batches
        epoch_cls_loss /= num_batches
        epoch_angle_loss /= num_batches
        epoch_iou_loss /= num_batches

        print(f"Train {epoch + 1}/{max_epoch} - "
              f'Mean loss: {epoch_loss:.4f}, '
              f'Cls loss: {epoch_cls_loss:.4f}, '
              f'Angle loss: {epoch_angle_loss:.4f}, '
              f'IoU loss: {epoch_iou_loss:.4f} | '
              f'Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')

        wandb.log({
            "Train Epoch loss": epoch_loss,
            "Train Cls loss": epoch_cls_loss,
            "Train Angle loss": epoch_angle_loss,
            "Train IoU loss": epoch_iou_loss,
            "Learning rate": optimizer.param_groups[0]["lr"]
        })
        ######################################################
        ##### Validation
        if validation:
            val_dataset = SceneTextDataset_val(split=val_dir, image_size=image_size, crop_size=input_size)
            val_dataset = EASTDataset(val_dataset)
            val_num_batches = math.ceil(len(val_dataset) / batch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

            ##### Validation Cycle
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()

                epoch_loss = 0
                epoch_start = time.time()

                epoch_cls_loss = 0.
                epoch_angle_loss = 0.
                epoch_iou_loss = 0.

                with tqdm(total=val_num_batches) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                        pbar.set_description(f'[Epoch {epoch + 1}]')

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                        loss_val = loss.item()
                        epoch_loss += loss_val

                        epoch_cls_loss += extra_info['cls_loss']
                        epoch_angle_loss += extra_info['angle_loss']
                        epoch_iou_loss += extra_info['iou_loss']

                        pbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }
                        pbar.set_postfix(val_dict)

                epoch_loss /= val_num_batches
                epoch_cls_loss /= val_num_batches
                epoch_angle_loss /= val_num_batches
                epoch_iou_loss /= val_num_batches

                print(f"Validation {epoch + 1}/{max_epoch} - "
                      f'Mean loss: {epoch_loss:.4f}, '
                      f'Best Validation loss: {best_val_loss:.4f}, | '
                      f'Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')

                wandb.log({
                    "Val Epoch loss": epoch_loss,
                    "Val Cls loss": epoch_cls_loss,
                    "Val Angle loss": epoch_angle_loss,
                    "Val IoU loss": epoch_iou_loss
                })

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)
                    ckpt_fpath = osp.join(model_dir, "best_val_loss.pth")
                    torch.save(model.state_dict(), ckpt_fpath)
                    cnt = 0
                    print(f"New Best Validation Loss at Epoch {epoch + 1}, Saving the Best Model to {ckpt_fpath}")
                else:
                    cnt += 1

                if cnt > early_stop_value:
                    break

        #######################################################
        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f"Model Checkpoint Saved at Epoch {epoch + 1} to '{ckpt_fpath}'")


def main(args):
    seed_everything(42)
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="Data-Annotation", entity="cv_19", config=args, name=args.exp_name)
    wandb.config.update(args)
    main(args)
