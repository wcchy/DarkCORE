import torch
import torch.nn as nn
from ...builder import DETECTORS, build_backbone, build_head, build_neck
from ..single_stage import SingleStageDetector
from ...backbones.DarkCORE.blocks import pair_downsampler, TextureDifference, LocalMean
from mmdet.core import bbox2result
import torch.nn.functional as F
import pdb

@DETECTORS.register_module()
class DarkCORE_YOLOV3(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 pre_encoder,
                 recon_loss_factor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DarkCORE_YOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)

        self.pre_encoder = build_backbone(pre_encoder)
        self.recon_loss_factor = int(recon_loss_factor)
        self.texture_diff = TextureDifference()
        self.local_mean = LocalMean()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_none = nn.MSELoss(reduction='none')

        print('recon_loss_factor:', self.recon_loss_factor)

        
    def reconstruct_loss(self, r, l, img):
        loss_recon = self.mse_loss(r * l, img)
        return loss_recon * self.recon_loss_factor
    
    def denoise_loss(self, down_1, down_2, denoi_pred1, denoi_pred2, deno_r):
        scale_1 = 1000
        scale_2 = 10000

        loss_denoise = self.mse_loss(down_1, denoi_pred2)* scale_1
        loss_denoise += self.mse_loss(down_2, denoi_pred1)* scale_1
        denoimg_1, denoimg_2 = pair_downsampler(deno_r)
        loss_denoise += self.mse_loss(denoi_pred1, denoimg_1)* scale_1
        loss_denoise += self.mse_loss(denoi_pred2, denoimg_2)* scale_1
        
        denoimg_1_denoimg_2_diff = self.texture_diff(denoimg_1, denoimg_2)
        denoimg_1_denoimg_2_diff = denoimg_1_denoimg_2_diff.clamp(0, 1)

        local_mean1 = self.local_mean(denoimg_1)
        local_mean2 = self.local_mean(denoimg_2)
        
        # weighted_diff1 = (1 - denoimg_1_denoimg_2_diff) * local_mean1 + denoimg_1 * denoimg_1_denoimg_2_diff
        # weighted_diff2 = (1 - denoimg_1_denoimg_2_diff) * local_mean2 + denoimg_2 * denoimg_1_denoimg_2_diff

        loss_area1 = self.mse_loss_none(denoimg_1, local_mean1)
        loss_area2 = self.mse_loss_none(denoimg_2, local_mean2)
        weighted_loss1 = (loss_area1 * scale_2 * (1 - denoimg_1_denoimg_2_diff)).mean()
        weighted_loss2 = (loss_area2 * scale_2 * (1 - denoimg_1_denoimg_2_diff)).mean()
        loss_denoise += (weighted_loss1 + weighted_loss2)

        return loss_denoise
    

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # pdb.set_trace()
        r, l, _, _, down_1, down_2, denoi_pred1, denoi_pred2, deno_r, x = self.pre_encoder(img)
        
        loss_decom = self.reconstruct_loss(r, l, img)
        loss_noise = self.denoise_loss(down_1, down_2, denoi_pred1, denoi_pred2, deno_r)

        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return loss_decom, loss_noise, x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        # print(img_metas)
        # print(img.shape)
        loss_decom, loss_noise, x = self.extract_feat(img)
    
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        
        losses['loss_decom'] = [loss_decom]
        losses['loss_noise'] = [loss_noise]

        return losses
    

    def simple_test(self, img, img_metas, rescale=False):
            """Test function without test-time augmentation.

            Args:
                img (torch.Tensor): Images with shape (N, C, H, W).
                img_metas (list[dict]): List of image information.
                rescale (bool, optional): Whether to rescale the results.
                    Defaults to False.

            Returns:
                list[list[np.ndarray]]: BBox results of each image and classes.
                    The outer list corresponds to each image. The inner list
                    corresponds to each class.
            """
            _, _, feat = self.extract_feat(img)
            # pdb.set_trace()
            results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        _, _, feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        _, _, x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels