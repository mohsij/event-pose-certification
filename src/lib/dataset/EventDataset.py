
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from core.certifier import Certifier
from core.pose_estimator import PoseEstimatorNoModel

from kornia.geometry.conversions import rotation_matrix_to_quaternion

logger = logging.getLogger(__name__)


class EventDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transforms_event=None):
        self.num_joints = 0
        self.pixel_std = 200

        self.is_train = is_train
        
        self.root = root
        
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.input_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform_event = transforms_event
        self.db = []
        
        self.cfg = cfg
        
        self.DATA_DIR = os.path.join(root, 'images')

        self.num_joints = cfg.MODEL.NUM_JOINTS

        self.image_width = cfg.DATASET.IMAGE_WIDTH
        self.image_height = cfg.DATASET.IMAGE_HEIGHT
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.db = self._get_db()
    
        device = 'cuda'
    
        self.landmarks_tensor = torch.tensor(self.landmarks, device=device, dtype=torch.float)
        self.intrinsics_tensor_event = torch.tensor(self.intrinsics_event, device=device, dtype=torch.float)
        self.pose_estimator_event = PoseEstimatorNoModel(cfg, self.intrinsics_tensor_event, self.landmarks_tensor)
        self.cad_model = torch.tensor(self.cad_model, device=device, dtype=torch.float)

        self.certifier = Certifier(cfg, self.intrinsics_tensor_event, self.landmarks_tensor, self.cad_model, cfg.TEST.BATCH_SIZE_PER_GPU)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, self.image_set+'.json'
        )

        with open(file_name, 'r') as anno_file:
            annotations = json.load(anno_file)

        self.landmarks = np.array(annotations["landmarks_3d"])
        self.intrinsics_event = np.array(annotations["intrinsics_event"])
        self.extrinsics = np.array(annotations["extrinsics_event_to_rgb"])
        
        dense_file_name = os.path.join(
            self.cfg.DATASET.ROOT, '..', 'dense-pointcloud', 'test.json'
        )
        with open(dense_file_name, 'r') as dense_file:
            self.cad_model = np.array(json.load(dense_file)["dense_pointcloud"])

        gt_db = []
        for a in annotations["annotations"]:
            image_name_event = a["filename_event"]

            bbox_event = np.array(a['bbox_event']).flatten()
            c_event, s_event = self._box2cs(bbox_event)

            joints_3d_event = np.zeros((self.num_joints, 3), dtype=float)
            joints_reshaped_event = np.array(a['keypoints_event']).reshape((-1,2))
            joints_3d_event[:, 0:2] = joints_reshaped_event[:, 0:2]

            # # Convert detectron visibility labels to mpii.
            # joints_3d_vis = np.zeros((self.num_joints,  3), dtype=float)
            # joints_3d_vis[:, 0] = joints_reshaped[:, -1] - 1
            # joints_3d_vis[:, 1] = joints_reshaped[:, -1] - 1

            image_dir = self.DATA_DIR

            gt_db.append(
                {
                    'image_event': os.path.join(image_dir, image_name_event),
                    'center_event': c_event,
                    'scale_event': s_event,  
                    'bbox_event': bbox_event,
                    'joints_3d_event': joints_3d_event,
                    'pose_event': np.array(a['pose_event']),
                    'filename_event': image_name_event,
                }
            )

        return gt_db

    def _box2cs(self, box):
        x1, y1, x2, y2 = box[:4]
        w = x2 - x1
        h = y2 - y1
        
        # make the bbox square otherwise the center and scale
        # ends up making the object out of bounds for rectangular bboxes.
        if w > h:
            h = w
        else:
            w = h
        
        return self._xywh2cs(x1, y1, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            # make every keypoint end up in the heatmap
            scale = scale * 1.2

        return center, scale

    def evaluate(self, cfg, output_dir, poses_gt_event, keypoint_predictions_event,  pose_predictions_event, filenames_event):
        pred_file = os.path.join(output_dir, '{}.json'.format("predictions"))
        total_translation_error_event = 0

        total_rotation_error_event = 0
        
        total_pose_error_event = 0
        
        all_translation_errors_event = []
        
        all_rotation_errors_event = []
        '''
            'full_image_path_event': db_rec['image_event'],
            'image_filename_event': db_rec['filename_event'],
            'joints_event': joints_event,
            'pose_event':db_rec['pose_event'],
            'bbox_event': db_rec['bbox_event'],
            'center_event': c_event,
            'scale_event': s_event,
            'rotation_event': r_event,
            'score_event': score_event
        '''
        annotations = []
        for i in range(len(filenames_event)):
            pose_gt_event = poses_gt_event[i]
            
            pose_pred_event = pose_predictions_event[i]
            
            rotation_gt_event = rotation_matrix_to_quaternion(torch.tensor(pose_gt_event[:3,:3])).cpu().numpy()
            translation_gt_event = pose_gt_event[:3, -1]
            
            rotation_pred_event = rotation_matrix_to_quaternion(torch.tensor(pose_pred_event[:3,:3])).cpu().numpy()
            translation_pred_event = pose_pred_event[:3, -1]
            
            translation_error_event = np.linalg.norm(translation_gt_event.flatten() - translation_pred_event.flatten(), 2) / np.linalg.norm(translation_gt_event.flatten())
            rotation_error_event = np.absolute(2 * np.arccos(np.absolute((rotation_gt_event * rotation_pred_event).sum())))
            
            total_rotation_error_event += rotation_error_event
            
            total_translation_error_event += translation_error_event
            
            all_translation_errors_event.append(translation_error_event)
            
            all_rotation_errors_event.append(rotation_error_event)
            
            total_pose_error_event += (translation_error_event + rotation_error_event)
            
            annotations.append({
                "filename_event":filenames_event[i],
                "pose_event":pose_predictions_event[i].tolist(),
                "keypoints_event": keypoint_predictions_event[i, :, :-1].tolist(),
                "rotation_error_event": str(rotation_error_event),
                "translation_error_event": str(translation_error_event),
                })
            
        # save the annotations in the same blender output format for easy compare
        with open(pred_file, 'w') as jsonfile:
            jsonfile.write(json.dumps({"annotations":annotations}, indent=2))

        total_translation_error_event /= len(filenames_event)
        
        total_rotation_error_event /= len(filenames_event)

        total_pose_error_event /= len(filenames_event)

        logger.info('event mean translation error {}'.format(total_translation_error_event))
        logger.info('event median translation error {}'.format(np.median(all_translation_errors_event)))
        logger.info('event mean rotation error {}'.format(total_rotation_error_event))
        logger.info('event median rotation error {}'.format(np.median(all_rotation_errors_event)))
        logger.info('event mean pose error {}'.format(total_pose_error_event))

        return total_pose_error_event

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        
        image_numpy_event = cv2.imread(
            db_rec['image_event'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:
            image_numpy_event = cv2.cvtColor(image_numpy_event, cv2.COLOR_BGR2RGB)
        
        if image_numpy_event is None:
            logger.error('=> fail to read {}'.format(db_rec['image_event']))
            raise ValueError('Fail to read {}'.format(db_rec['image_event']))
        
        joints_event = db_rec['joints_3d_event']
        c_event = db_rec['center_event']
        s_event = db_rec['scale_event']
        score_event = 1
        r_event = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            
            scale_multiplier = np.clip(np.random.randn()*sf + 1, 1.0, 1 + sf)
            s_event = s_event * scale_multiplier
            
            random_rotation = np.clip(np.random.randn()*rf, -rf*2, rf*2)
            dice_roll = random.random()
            r_event = random_rotation if dice_roll <= 0.6 else 0
        
        trans_event = get_affine_transform(c_event, s_event, r_event, self.input_size)
        trans_hm_to_img_event = get_affine_transform(c_event, s_event, 0, self.heatmap_size, inv=1)
        trans_crop_to_img_event = get_affine_transform(c_event, s_event, 0, self.input_size, inv=1)
        
        input_tensor_event = cv2.warpAffine(
            image_numpy_event,
            trans_event,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR)


        if self.transform_event:
            input_tensor_event = self.transform_event(input_tensor_event)

        for i in range(self.num_joints):
            joints_event[i, 0:2] = affine_transform(joints_event[i, 0:2], trans_event)
        
        target_event, target_weight_event = self.generate_target(joints_event)
        target_event = torch.from_numpy(target_event)
        target_weight_event = torch.from_numpy(target_weight_event)

        meta = {
            'full_image_path_event': db_rec['image_event'],
            'image_filename_event': db_rec['filename_event'],
            'joints_event': joints_event,
            'pose_event':db_rec['pose_event'],
            'bbox_event': db_rec['bbox_event'],
            'center_event': c_event,
            'scale_event': s_event,
            'rotation_event': r_event,
            'score_event': score_event
        }
        
        trans_hm_to_img_event =  torch.tensor(trans_hm_to_img_event, device='cuda', dtype=torch.float)
        trans_crop_to_img_event = torch.tensor(trans_crop_to_img_event, device='cuda', dtype=torch.float) 

        return input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, meta

    # def select_data(self, db):
    #     db_selected = []
    #     for rec in db:
    #         num_vis = 0
    #         joints_x = 0.0
    #         joints_y = 0.0
    #         for joint, joint_vis in zip(
    #                 rec['joints_3d'], rec['joints_3d_vis']):
    #             if joint_vis[0] <= 0:
    #                 continue
    #             num_vis += 1

    #             joints_x += joint[0]
    #             joints_y += joint[1]
    #         if num_vis == 0:
    #             continue

    #         joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

    #         area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
    #         joints_center = np.array([joints_x, joints_y])
    #         bbox_center = np.array(rec['center'])
    #         diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
    #         ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

    #         metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
    #         if ks > metric:
    #             db_selected.append(rec)

    #     logger.info('=> num db: {}'.format(len(db)))
    #     logger.info('=> num selected db: {}'.format(len(db_selected)))
    #     return db_selected

    def generate_target(self, joints, heatmap_divide=1):
        '''
        :param joints:  [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            heatmap_size = (self.heatmap_size/heatmap_divide).astype('int')
            if heatmap_divide==1:
                sigma = self.sigma
            elif heatmap_divide==2:
                sigma = self.sigma2
            elif heatmap_divide==4:
                sigma = self.sigma3
            elif heatmap_divide==8:
                sigma = self.sigma4
            else:
                assert False, 'heatmap scales out of predefined range.'
            
            target = np.zeros((self.num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.input_size / heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
