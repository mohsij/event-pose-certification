import torch
import cv2
import numpy as np

from torchvision.utils import draw_keypoints, draw_bounding_boxes
from torchvision.io import write_png, read_image, ImageReadMode
from torchvision.ops import generalized_box_iou_loss

from utils.BPnP import BPnP
from utils.BPnP import batch_project
from utils.utils import get_events_from_input, floor_with_gradient

from kornia.geometry.subpix.spatial_soft_argmax import SpatialSoftArgmax2d
from kornia.geometry.conversions import axis_angle_to_rotation_matrix

from core.inference import get_final_preds
from utils.transforms import get_affine_transform

from pytorch3d import ops
from pytorch3d.loss import chamfer_distance

img_iter = 0

class Corrector:
    def __init__(self, cfg, model, K, landmarks, cad_model):
        '''
        K: Tensor containing camera intrinsics of shape (3, 3)
        landmarks: Tensor of shape (N, 3)
        cad_model: Tuple of 2 Tensors of shape (m1, 3) and (m2, 3) : point cloud representation
        '''
        self.model = model
        self.cfg = cfg
        self.K = K
        self.landmarks = landmarks
        self.bpnp = BPnP.apply
        self.floor = floor_with_gradient.apply
        self.cad_model = cad_model

    def solve(self, pred_keypoints, keypoint_scores, input_batch, meta, trans_crop_to_img):
        """
        pre_keypoints   : Tensor of shape (b, num_kp, 2)
        input           : Tensor of shape (b, 3, w, h)

        """
        device = 'cuda'
        input_batch = input_batch.to(device)

        all_corrections = self.batch_gradient_descent(pred_keypoints, keypoint_scores, input_batch, meta, trans_crop_to_img)

        return all_corrections, None

    def objective(self, preds_keypoints, keypoint_scores, input_batch, meta, trans_crop_to_img, correction):
        """
        pred_corrected  : Tensor of shape (b, num_kp, 2)
        input           : Tensor of shape (b, 3, w, h)
        correction      : Tensor of shape (b, num_kp, 2)

        Outputs:
        obj             : Tensor of shape (b, 1)

        """

        #breakpoint()

        # :TODO: Make these parameters of the class
        sigma_1 = 1.0
        sigma_2 = 1.0
        sigma_3 = 1.0

        batch_size = input_batch.shape[0]

        device = self.K.device

        poses_predicted = self.bpnp(preds_keypoints + correction, self.landmarks, self.K, keypoint_scores)

        # Keypoint loss:
        reprojected_landmarks = batch_project(poses_predicted, self.landmarks, self.K)
        corrected_landmarks = preds_keypoints + correction
        intermediate = torch.pow(corrected_landmarks - reprojected_landmarks.float(), 2).sum(dim=2)
        keypoint_loss = (intermediate * keypoint_scores.squeeze(-1)).mean(dim=1)

        dataset_cfg = self.cfg.DATASET
        resolution = (dataset_cfg.IMAGE_WIDTH, dataset_cfg.IMAGE_HEIGHT)

        points_2d = batch_project(poses_predicted, self.cad_model, self.K, clamp_to_resolution=resolution)

        events_batch, events_counts = get_events_from_input(input_batch, trans_crop_to_img)
        global img_iter

        img_iter += 1

        mask_loss, _ = chamfer_distance(
            events_batch, 
            points_2d,
            x_lengths=events_counts.to(device).long(),
            batch_reduction=None,
            single_directional=True
        )
        # print(obj)
        for count, i in enumerate(events_batch):
            projected_image = torch.zeros((3, resolution[1], resolution[0]), dtype=torch.uint8, device=device)
            projected_image = draw_keypoints(projected_image, i.unsqueeze(0), colors='#00FFFF', radius=1)
            #segmentation mask is green
            #projected_image = draw_keypoints(projected_image, points_2d[count].unsqueeze(0), colors='#00FF00', radius=1)
            projected_image = draw_keypoints(projected_image, corrected_landmarks[count].unsqueeze(0), colors='#FF00FF', radius=3)
            projected_image = draw_keypoints(projected_image, reprojected_landmarks[count].unsqueeze(0), colors='#FFFFFF', radius=3)
            write_png(projected_image, 'test/events_{}_{}.png'.format(count, img_iter))
        return sigma_1 * mask_loss.reshape((-1,1)) + sigma_2 * keypoint_loss.reshape((-1,1))

    def batch_gradient_descent(self, pred_keypoints, keypoint_scores, input_batch, meta, trans_crop_to_img, lr=1, max_iterations=500, tol=1e-12):
        """
        inputs:
        pred_keypoints      : Tensor of shape (b, num_kp, 2)
        input               : Tensor of shape (b, 3, w, h)

        outputs:
        correction          : Tensor of shape (b, num_kp, 2)
        """

        def _get_objective_jacobian(fun, x):

            torch.set_grad_enabled(True)
            batch_size = x.shape[0]
            dfdcorrection = torch.zeros_like(x)

            # Do not set create_graph=True in jacobian. It will slow down computation substantially.
            dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
            b = range(batch_size)
            # print('gradients')
            # print(dfdcorrectionX)
            dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

            return dfdcorrection

        num_kp = pred_keypoints.shape[1]
        correction = torch.zeros_like(pred_keypoints)

        f = lambda x: self.objective(pred_keypoints, keypoint_scores, input_batch, meta, trans_crop_to_img, x)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr

        iter = 0
        obj_ = f(correction)
        flag = torch.ones_like(obj_).to(dtype=torch.bool)
        # flag_idx = flag.nonzero()
        flag = flag.unsqueeze(-1).repeat(1, num_kp, 2)
        while iter < max_iterations:

            iter += 1
            obj = obj_

            dfdcorrection = _get_objective_jacobian(f, correction)
            correction -= lr*dfdcorrection*flag

            obj_ = f(correction)

            if (obj-obj_).abs().max() < tol:

                break
            else:
                flag = (obj-obj_).abs() > tol
                # flag_idx = flag.nonzero()
                flag = flag.unsqueeze(-1).repeat(1, num_kp, 2)

        return correction

    def gradient(self, pred_keypoints, input_batch, y=None, v=None, ctx=None):
        if v is None:
            v = torch.ones_like(pred_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for pred_keypoints.
        # We don't backpropagate gradient with respect to the input. Therefore, None value in the second tuple.
        return (-v, None)