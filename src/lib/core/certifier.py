'''
python=3.9.18
torch=2.3.1
numpy=2.0.2
scipy=1.13.1
matplotlib=3.1.3
cuda=12.1
'''

import torch
import cv2
import numpy as np

from torchvision.utils import draw_keypoints, draw_bounding_boxes
from pytorch3d.loss import chamfer_distance
from torchvision.io import write_png, read_image, ImageReadMode

from utils.BPnP import batch_project
from utils.utils import get_events_from_input, floor_with_gradient

class Certifier:
    def __init__(self, cfg, K, landmarks, cad_model, batch_size):
        self.cfg = cfg
        self.K = K
        self.landmarks = landmarks
        self.cad_model = cad_model
        self.floor = floor_with_gradient.apply
        self.batch_size = batch_size

    def certify(self, pred_pose, pred_landmarks, input_batch, meta, trans_crop_to_img, epsilon=100.0, is_val=True, epoch=0, batch_count=0):
        with torch.no_grad():

            device = self.K.device

            dataset_cfg = self.cfg.DATASET
            resolution = (dataset_cfg.IMAGE_WIDTH, dataset_cfg.IMAGE_HEIGHT)

            points_2d = batch_project(pred_pose, self.cad_model, self.K)

            events_batch, events_counts = get_events_from_input(input_batch, trans_crop_to_img)
            
            event_to_mask_hausdorff_dist, _ = chamfer_distance(
                events_batch, 
                points_2d,
                x_lengths=events_counts.to(device).long(),
                point_reduction=None,
                batch_reduction=None,
                single_directional=True
            )

            event_counts_threshold_mask = (events_counts.to(device).long() > 100)

            dataset_cfg = self.cfg.DATASET
            resolution = (dataset_cfg.IMAGE_WIDTH, dataset_cfg.IMAGE_HEIGHT)

            dist_with_90_percent_points = event_to_mask_hausdorff_dist.quantile(0.9997, 1).reshape((-1, 1))

            event_to_mask_hausdorff_dist = event_to_mask_hausdorff_dist * (event_to_mask_hausdorff_dist < dist_with_90_percent_points)

            event_to_mask_hausdorff_dist = event_to_mask_hausdorff_dist.amax(1)

            # print("dist:", event_to_mask_hausdorff_dist)

            certification, certification_scores = event_to_mask_hausdorff_dist < epsilon, event_to_mask_hausdorff_dist
            certification = torch.logical_and(certification, event_counts_threshold_mask)

            # for count, i in enumerate(events_batch):
            #     projected_image_cad = torch.full((3, resolution[1], resolution[0]), 255, dtype=torch.uint8, device=device)
            #     projected_image_events = torch.full((3, resolution[1], resolution[0]), 255, dtype=torch.uint8, device=device)
            #     projected_image_both = torch.full((3, resolution[1], resolution[0]), 255, dtype=torch.uint8, device=device)
                
            #     try:
            #         if is_val:
            #             cad_colour = '#00EE00' if certification[count] else '#EE0000'
            #             # # events are cyan
            #             # projected_image_events = draw_keypoints(projected_image_events, events_batch[count].unsqueeze(0), colors='#000000', radius=1)
            #             # # #segmentation mask is green
            #             # projected_image_cad = draw_keypoints(projected_image_cad, points_2d[count].unsqueeze(0), colors=cad_colour, radius=1)
                        
            #             #segmentation mask is green
            #             projected_image_both = draw_keypoints(projected_image_both, points_2d[count].unsqueeze(0), colors=cad_colour, radius=1)
            #             # events are white
            #             projected_image_both = draw_keypoints(projected_image_both, events_batch[count].unsqueeze(0), colors='#000000', radius=1)

            #             # write_png(projected_image_events, 'certifier_video/events/{}_{}.png'.format(epoch, (batch_count*self.batch_size) + count))
            #             # write_png(projected_image_cad, 'certifier_video/cad/{}_{}.png'.format(epoch, (batch_count*self.batch_size) + count))
            #             write_png(projected_image_both, 'certifier_video/both/{}_{}.png'.format(epoch, (batch_count*self.batch_size) + count))
            #     except SystemError:
            #         pass

            return certification, certification_scores

