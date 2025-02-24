
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate_certifier
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from utils.transforms import EventNormalise, FillEventBlack, RandomEventNoise, RandomEventPatchNoise

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    hrnet_event = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    
    hrnet_encoder_event = eval('models.'+cfg.MODEL.NAME+'.get_encoder')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model_event = torch.nn.DataParallel(hrnet_event, device_ids=cfg.GPUS).cuda()
    model_hrnet_encoder_event = torch.nn.DataParallel(hrnet_encoder_event, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    heatmap_loss_event = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    valid_dataset_event = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, is_train=False,
        transforms_event=transforms.Compose([
            transforms.ToTensor(),
            FillEventBlack(),
        ]),
    )
    
    valid_loader_event = torch.utils.data.DataLoader(
        valid_dataset_event,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = cfg.TEST.MODEL_FILE

    checkpoint_file_event = checkpoint_file.replace("checkpoint", "checkpoint_event")

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file_event):
        logger.info("=> loading event checkpoint '{}'".format(checkpoint_file_event))
        checkpoint = torch.load(checkpoint_file_event)
        model_event.load_state_dict(checkpoint['state_dict_event'])
        model_hrnet_encoder_event.load_state_dict(checkpoint['state_dict_encoder_event'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file_event, checkpoint['epoch']))

    # evaluate on validation set
    validate_certifier(
        cfg, valid_loader_event, valid_dataset_event,
        model_event,
        hrnet_encoder_event,
        heatmap_loss_event,
        final_output_dir, tb_log_dir, writer_dict=writer_dict
    )

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
