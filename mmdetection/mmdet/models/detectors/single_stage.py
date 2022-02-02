import torch.nn as nn
import torch
from mmdet.core import bbox2result, bbox2roi
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from mmdet.models.reid_head.reid import build_reid

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 bbox_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        if test_cfg.with_reid:
            self.reid_head = build_reid(test_cfg)
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        if self.test_cfg.with_reid:
            self.bbox_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        if img_metas == 'use_mr':
            x = self.extract_feat(img)
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], bbox2roi(gt_bboxes))
            feats = self.reid_head(bbox_feats)
            return feats

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        cls_labels = [i[:, 0] for i in gt_labels]
        loss_inputs = outs + (gt_bboxes, cls_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if not self.train_cfg.with_reid:  # detection
            return losses
        else:  # person search
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], bbox2roi(gt_bboxes))
            feats = self.reid_head(bbox_feats)
            return losses, feats

    def simple_test(self, img, img_meta, rescale=False, gt_box=None):
        # person search -- query
        if gt_box is not None:
            x = self.extract_feat(img)
            gt_bbox_list = gt_box[0][0]
            gt_bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], bbox2roi([gt_bbox_list]))
            gt_bbox_feats = self.reid_head(gt_bbox_feats)
            gt_bbox_list = torch.cat([gt_bbox_list / img_meta[0]['scale_factor'],
                                      torch.ones(gt_bbox_list.shape[0], 1).cuda()], dim=-1)
            bbox_results = [bbox2result(gt_bbox_list, torch.zeros(gt_bbox_list.shape[0]), self.bbox_head.num_classes)]
            return bbox_results, gt_bbox_feats.cpu().numpy()

        # person search -- gallery
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        if self.test_cfg.with_reid:
            pre_bbox_list = bbox_list[0][0] * img_meta[0]['scale_factor']
            pre_bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], bbox2roi([pre_bbox_list]))
            pre_bbox_feats = self.reid_head(pre_bbox_feats)
            return bbox_results, pre_bbox_feats.cpu().numpy()

        # detection
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
