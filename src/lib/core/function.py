
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images_certifier, save_debug_images_certifier_test
from kornia.geometry.conversions import axis_angle_to_rotation_matrix

from utils.BPnP import batch_project

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import transform_preds

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

logger = logging.getLogger(__name__)

def train_certifier(config, 
                    train_loader_event_real, train_dataset_event_real,
                    model_event, model_hrnet_encoder_event,
                    heatmap_loss_event,
                    optimizer_event,
                    epoch, 
                    certifier_eps,
                    output_dir, tb_log_dir, writer_dict, original_eps=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()


    end = time.time()
    total_certified_for_epoch = 0
    for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, meta) in enumerate(train_loader_event_real):        
        # measure data loading time
        data_time.update(time.time() - end)
        
        model_event.eval()
        model_hrnet_encoder_event.eval()
        
        features_event = model_hrnet_encoder_event(input_tensor_event)
        outputs_event = model_event(features_event)
        pred_keypoints, poses, keypoint_scores = train_dataset_event_real.pose_estimator_event.predict(outputs_event.clone(), trans_hm_to_img_event)
        certified_indices, certification_scores = train_dataset_event_real.certifier.certify(poses, pred_keypoints, input_tensor_event, meta, trans_crop_to_img_event, epsilon=certifier_eps, is_val=False)
        if original_eps is not None:
            certified_indices_with_original_eps, _ = train_dataset_event_real.certifier.certify(poses, pred_keypoints, input_tensor_event, meta, trans_crop_to_img_event, epsilon=original_eps, is_val=False)
            total_certified_for_epoch += certified_indices_with_original_eps.sum()
        if certified_indices.sum() > 0:
            certified_inputs = input_tensor_event[certified_indices.cpu()]
            certified_poses = poses[certified_indices.cpu()]
            certified_keypoints = pred_keypoints[certified_indices.cpu()]

            plot_joints = []
            corrected_landmarks = batch_project(certified_poses, train_dataset_event_real.landmarks_tensor, train_dataset_event_real.intrinsics_tensor_event)
            
            model_event.train()
            model_hrnet_encoder_event.train()
            
            # compute output
            certified_features_event = model_hrnet_encoder_event(certified_inputs)
            
            certified_outputs_event = model_event(certified_features_event)
        
            c = meta['center_event'][certified_indices.cpu()].cpu().numpy()
            s = meta['scale_event'][certified_indices.cpu()].cpu().numpy()
            
            # target shape should be (batch, num_joints, hm_width, hm_height)
            # target_weight shape should be (batch, num_joints, 1)
            new_target = []
            new_target_weight = []
            # print("shapes")
            # print(target.shape)
            # print(target_weight.shape)
            # print("gathering...")
            for certified_instance_count, center in enumerate(c):
                scale = s[certified_instance_count]
                trans_img_to_hm = get_affine_transform(center, scale, 0, train_dataset_event_real.input_size)
                joints = corrected_landmarks[certified_instance_count].cpu().numpy()
                for joint_count in range(train_dataset_event_real.num_joints):
                    joints[joint_count] = affine_transform(joints[joint_count], trans_img_to_hm)
                plot_joints.append(joints)
                #joints_vis = np.column_stack((np.ones_like(joints), np.zeros((train_dataset_event_real.num_joints, 1))))
                new_hm, new_hm_weight = train_dataset_event_real.generate_target(joints)
                new_target.append(torch.Tensor(new_hm).unsqueeze(0))
                new_target_weight.append(torch.Tensor(new_hm_weight).unsqueeze(0))
            # print(new_target)
            new_target = torch.cat(new_target, 0)
            new_target_weight = torch.cat(new_target_weight, 0)

            new_target = new_target.cuda(non_blocking=True)
            new_target_weight = new_target_weight.cuda(non_blocking=True)

            # print("new shapes")
            # print(new_target.shape)
            # print(new_target_weight.shape)

            loss = heatmap_loss_event(certified_outputs_event, new_target, new_target_weight)

            # compute gradient and do update step
            optimizer_event.zero_grad()
            loss.backward()
            optimizer_event.step()

            # measure accuracy and record loss
            _, avg_acc, cnt, pred_event = accuracy(certified_outputs_event.detach().cpu().numpy(),
                                                   new_target.detach().cpu().numpy())
            acc_event.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss_event {loss_event.val:.7f} ({loss_event.avg:.7f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(train_loader_event_real), batch_time=batch_time,
                        speed=input_tensor_event.size(0)/batch_time.val,
                        data_time=data_time, loss_event=losses_event, acc=acc_event)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss_event', losses_event.val, global_steps)
                writer.add_scalar('train_acc_event', acc_event.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
                
                new_meta = {}
                # print(certified_indices)
                for key in meta:
                    if isinstance(meta[key], list):
                        new_meta[key] = []
                        for index, certified_index in enumerate(certified_indices.cpu().numpy()):
                            if certified_index == True:
                                new_meta[key].append(meta[key][index])
                        # print(new_meta[key])
                    else:
                        new_meta[key] = meta[key][certified_indices.cpu()]

                prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)

                save_debug_images_certifier(
                    config, 
                    certified_inputs,
                    new_meta,
                    new_target,
                    pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])),
                    plot_joints*4,
                    certified_outputs_event,
                    prefix)
    print("total certified: {}".format(total_certified_for_epoch))

def validate_certifier(
    config, 
    val_loader, val_dataset,
    model_event, model_hrnet_encoder_event, 
    heatmap_loss_event, 
    output_dir, tb_log_dir, writer_dict=None, epoch=0):
    
    batch_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()

    # switch to evaluate mode
    model_event.eval()
    model_hrnet_encoder_event.eval()

    num_samples = len(val_dataset)
    
    keypoint_predictions_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    # 4x4 poses
    pose_predictions_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_event = []

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, meta) in enumerate(val_loader):
            # compute output
            input_tensor_event = input_tensor_event.cuda()
            features_event = model_hrnet_encoder_event(input_tensor_event)
            output_event = model_event(features_event)

            target_event = target_event.cuda(non_blocking=True)
            target_weight_event = target_weight_event.cuda(non_blocking=True)

            loss = heatmap_loss_event(output_event, target_event, target_weight_event)

            num_images = input_tensor_event.size(0)
            # measure accuracy and record loss
            losses_event.update(loss.item(), num_images)
            _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                             target_event.detach().cpu().numpy())

            acc_event.update(avg_acc, cnt)

            centre_event = meta['center_event'].numpy()
            scale_event = meta['scale_event'].numpy()

            preds_event, maxvals = get_final_preds(
                config, output_event.clone().cpu().numpy(), centre_event, scale_event)

            keypoint_predictions_event[idx:idx + num_images, :, 0:2] = preds_event[:, :, 0:2]
            keypoint_predictions_event[idx:idx + num_images, :, 2:3] = maxvals
            
            filenames_event.extend(meta['image_filename_event'])
            
            # get the event poses and add to the all poses predictions
            keypoints_bpnp_event, poses_bpnp_event, _ = val_dataset.pose_estimator_event.predict(output_event.clone(), trans_hm_to_img_event)
            poses_bpnp_event = poses_bpnp_event.clone()
            pose_predictions_event[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_event[:, 0:3]).cpu().numpy()
            poses_bpnp_event = poses_bpnp_event.cpu().numpy()
            pose_predictions_event[idx:idx + num_images, 0:3, -1] = poses_bpnp_event[:, 3:].reshape((-1, 3))
            
            pose_gt_event[idx:idx + num_images, :, :] = meta["pose_event"]
            
            idx += num_images
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Testevent: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_event, acc=acc_event)
                logger.info(msg)

                prefix = '{}_{}_{}'.format(
                    os.path.join(output_dir, 'validation'), epoch, i
                )
                save_debug_images_certifier_test(
                    config, 
                    input_tensor_event, meta, 
                    pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    prefix)
                

        perf_indicator_event = val_dataset.evaluate(
            config, 
            output_dir,
            pose_gt_event,
            keypoint_predictions_event, 
            pose_predictions_event,
            filenames_event
        )

    return perf_indicator_event

#### no certifier initial synthetic training

def train(config, 
          train_loader_event, 
          model_event, model_hrnet_encoder_event,
          heatmap_loss_event,
          optimizer_event,
          epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()

    # switch to train mode
    model_event.train()
    
    model_hrnet_encoder_event.train()

    end = time.time()
    for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, meta) in enumerate(train_loader_event):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the features
        features_event = model_hrnet_encoder_event(input_tensor_event)
        
        # compute output from features
        outputs_event = model_event(features_event)
        
        target_event = target_event.cuda(non_blocking=True)
        target_weight_event = target_weight_event.cuda(non_blocking=True)

        output_event = outputs_event
        
        loss_event = heatmap_loss_event(output_event, target_event, target_weight_event)
        
        # compute gradient and do update step
        optimizer_event.zero_grad()

        loss_event.backward()
        
        optimizer_event.step()

        # measure accuracy and record loss
        losses_event.update(loss_event.item(), input_tensor_event.size(0))

        _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                               target_event.detach().cpu().numpy())
        acc_event.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_event {loss_event.val:.7f} ({loss_event.avg:.7f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader_event), batch_time=batch_time,
                      speed=input_tensor_event.size(0)/batch_time.val,
                      data_time=data_time, loss_event=losses_event, acc=acc_event)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_event', losses_event.val, global_steps)
            writer.add_scalar('train_acc_event', acc_event.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_debug_images_certifier_test(
                config, 
                input_tensor_event,
                meta, 
                pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), prefix)


# def validate(
#     config, 
#     val_loader, 
#     val_dataset, 
#     model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb, 
#     heatmap_loss_event, heatmap_loss_rgb, 
#     output_dir, tb_log_dir, writer_dict=None, epoch=0):
    
#     batch_time = AverageMeter()
#     losses_event = AverageMeter()
#     acc_event = AverageMeter()
#     losses_rgb = AverageMeter()
#     acc_rgb = AverageMeter()

#     # switch to evaluate mode
#     model_event.eval()
#     model_rgb.eval()
#     model_hrnet_encoder_event.eval()
#     model_hrnet_encoder_rgb.eval()

#     num_samples = len(val_dataset)
#     keypoint_predictions_event = np.zeros(
#         (num_samples, config.MODEL.NUM_JOINTS, 3),
#         dtype=np.float32
#     )
#     # 4x4 poses
#     pose_predictions_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
#     pose_gt_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
#     filenames_event = []
    
#     keypoint_predictions_rgb = np.zeros(
#         (num_samples, config.MODEL.NUM_JOINTS, 3),
#         dtype=np.float32
#     )
#     # 4x4 poses
#     pose_predictions_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
#     pose_gt_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
#     filenames_rgb = []

#     idx = 0
#     with torch.no_grad():
#         end = time.time()
#         for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(val_loader):
#             # compute output
#             input_tensor_event = input_tensor_event.cuda()
#             features_event = model_hrnet_encoder_event(input_tensor_event)
            
#             input_tensor_rgb = input_tensor_rgb.cuda()
#             features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
            
#             output_event = model_event(features_event)

#             target_event = target_event.cuda(non_blocking=True)
#             target_weight_event = target_weight_event.cuda(non_blocking=True)

#             loss = heatmap_loss_event(output_event, target_event, target_weight_event)

#             num_images = input_tensor_event.size(0)
#             # measure accuracy and record loss
#             losses_event.update(loss.item(), num_images)
#             _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
#                                              target_event.detach().cpu().numpy())

#             acc_event.update(avg_acc, cnt)
            
#             centre_event = meta['center_event'].numpy()
#             scale_event = meta['scale_event'].numpy()

#             preds_event, maxvals = get_final_preds(
#                 config, output_event.clone().cpu().numpy(), centre_event, scale_event)

#             keypoint_predictions_event[idx:idx + num_images, :, 0:2] = preds_event[:, :, 0:2]
#             keypoint_predictions_event[idx:idx + num_images, :, 2:3] = maxvals
            
#             filenames_event.extend(meta['image_filename_event'])
            
#             # get the event poses and add to the all poses predictions
#             keypoints_bpnp_event, poses_bpnp_event, _ = val_dataset.pose_estimator_event.predict(output_event.clone(), trans_hm_to_img_event)
#             poses_bpnp_event = poses_bpnp_event.clone()
#             pose_predictions_event[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_event[:, 0:3]).cpu().numpy()
#             poses_bpnp_event = poses_bpnp_event.cpu().numpy()
#             pose_predictions_event[idx:idx + num_images, 0:3, -1] = poses_bpnp_event[:, 3:].reshape((-1, 3))
            
#             pose_gt_event[idx:idx + num_images, :, :] = meta["pose_event"]
            
#             # evaluate rgb outputs
            
#             output_rgb = model_rgb(features_rgb)

#             target_rgb = target_rgb.cuda(non_blocking=True)
#             target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

#             loss = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)

#             num_images = input_tensor_rgb.size(0)
#             # measure accuracy and record loss
#             losses_rgb.update(loss.item(), num_images)
#             _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
#                                              target_rgb.detach().cpu().numpy())

#             acc_rgb.update(avg_acc, cnt)

#             centre_rgb = meta['center_rgb'].numpy()
#             scale_rgb = meta['scale_rgb'].numpy()

#             preds_rgb, maxvals = get_final_preds(
#                 config, output_rgb.clone().cpu().numpy(), centre_rgb, scale_rgb)

#             keypoint_predictions_rgb[idx:idx + num_images, :, 0:2] = preds_rgb[:, :, 0:2]
#             keypoint_predictions_rgb[idx:idx + num_images, :, 2:3] = maxvals
            
#             filenames_rgb.extend(meta['image_filename_rgb'])
            
#             # get the rgb poses and add to the all poses predictions
#             keypoints_bpnp_rgb, poses_bpnp_rgb, _ = val_dataset.pose_estimator_rgb.predict(output_rgb.clone(), trans_hm_to_img_rgb)
#             poses_bpnp_rgb = poses_bpnp_rgb.clone()
#             pose_predictions_rgb[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_rgb[:, 0:3]).cpu().numpy()
#             poses_bpnp_rgb = poses_bpnp_rgb.cpu().numpy()
#             pose_predictions_rgb[idx:idx + num_images, 0:3, -1] = poses_bpnp_rgb[:, 3:].reshape((-1, 3))
            
#             pose_gt_rgb[idx:idx + num_images, :, :] = meta["pose_rgb"]
            
#             idx += num_images
            
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % config.PRINT_FREQ == 0:
#                 msg = 'Testevent: [{0}/{1}]\t' \
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
#                       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
#                           i, len(val_loader), batch_time=batch_time,
#                           loss=losses_event, acc=acc_event)
#                 logger.info(msg)
#                 msg = 'Testrgb: [{0}/{1}]\t' \
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
#                       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
#                           i, len(val_loader), batch_time=batch_time,
#                           loss=losses_rgb, acc=acc_rgb)
#                 logger.info(msg)

#                 prefix = '{}_{}_{}'.format(
#                     os.path.join(output_dir, 'validation'), epoch, i
#                 )
#                 save_debug_images_certifier_test(
#                     config, 
#                     input_tensor_event, 
#                     input_tensor_rgb, meta, 
#                     pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
#                     pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
#                     prefix)
                

#         perf_indicator_event, perf_indicator_rgb = val_dataset.evaluate(
#             config, 
#             output_dir,
#             pose_gt_event,
#             keypoint_predictions_event, 
#             pose_predictions_event,
#             filenames_event,
#             pose_gt_rgb,
#             keypoint_predictions_rgb, 
#             pose_predictions_rgb,
#             filenames_rgb
#         )


#     return perf_indicator_event, perf_indicator_rgb
