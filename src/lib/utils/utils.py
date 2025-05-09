# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from torchvision.transforms.functional import rgb_to_grayscale

class floor_with_gradient(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def get_events_from_input(input_batch, trans_crop_to_img):
    """
    Gets 2D coordinates of white (event) points from the input image tensor.
    The output 2D coordinates are in full resolution image space i.e cfg.MODEL.IMAGE_SIZE
    """
    device = 'cuda'

    input_batch = input_batch.to(device)

    events_batch = []
    for image_index, i in enumerate(rgb_to_grayscale(input_batch)):
        # Try to cull out noise by only keeping pixel positions that experience more events (brighter white)
        coords = torch.argwhere(i.squeeze(0)>0.85)
        coords = torch.flip(coords, dims=(1,))
        events_batch.append(coords.float())

    event_counts = torch.zeros(len(events_batch))

    for i, events in enumerate(events_batch):
        new_events = torch.matmul(events, torch.tensor([[1.,0.,0.],[0.,1.,0.]], device='cuda', dtype=torch.float32))
        new_events += torch.tensor([0.,0.,1.], device='cuda', dtype=torch.float32)
        trans_torch = trans_crop_to_img[i]
        new_events = torch.matmul(new_events, trans_torch.T)
        events_batch[i] = torch.floor(new_events)
        event_counts[i] = new_events.shape[0]

    return torch.nn.utils.rnn.pad_sequence(events_batch, batch_first=True), event_counts


def project_points_in_2D(K, RT, points, resolution_px):
    """
    Project all 3D triangle vertices in the mesh into
    the 2D image of given resolution

    Parameters
    ----------
    K: ndarray
        Camera intrinsics matrix, 3x3
    RT: ndarray
        Camera pose (inverse of extrinsics), 3x4
    mesh: ndarray
        Triangles to be projected in 2d, (Nx3x3) 
    resolution_px: tuple
        Resolution of image in pixel

    Returns
    -------
    coords_projected_2D: ndarray
        Triangle vertices projected in 2D and clipped to
        image resolution
    """
    resolution_x_px, resolution_y_px = resolution_px  # image resolution in pixels

    # Correct reference system of extrinsics matrix
    #   y is down: (to align to the actual pixel coordinates used in digital images)
    #   right-handed: positive z look-at direction
    correction_factor = torch.from_numpy(np.asarray([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                                                    dtype=np.float32)).to(RT.device)

    RT = correction_factor @ RT.float()

    # Compose whole camera projection matrix (3x4)
    P = K @ RT

    len_points_flat = points.size(0)

    # Create constant tensor to store 3D model coordinates
    ones = torch.ones(len_points_flat, 1).to(RT.device)
    coords_3d_h = torch.cat([points, ones], dim=-1)  # n_triangles, 4
    coords_3d_h = coords_3d_h.t()                       # 4, n_triangles

    # Project 3D vertices into 2D
    coords_projected_2D_h = (P @ coords_3d_h).t()         # n_triangles, 3
    coords_projected_2D = coords_projected_2D_h[:, :2] / (coords_projected_2D_h[:, 2:] + 1e-8)

    # Clip indexes in image range (off by 1 pixel each side to avoid edge issues)
    coords_projected_2D_x_clip = torch.clamp(coords_projected_2D[:, 0: 1], -1, resolution_x_px)
    coords_projected_2D_y_clip = torch.clamp(coords_projected_2D[:, 1: 2], -1, resolution_y_px)
    return torch.cat([coords_projected_2D_x_clip, coords_projected_2D_y_clip], dim=-1)

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details
