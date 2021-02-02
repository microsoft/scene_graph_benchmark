# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
"""
Implements the FRCNN with Attribute Head
"""
import numpy as np
import torch

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import \
    GeneralizedRCNN
from .attribute_head.attribute_head import build_roi_attribute_head


class AttrRCNN(GeneralizedRCNN):
    """
    Main class for Generalized Relation R-CNN.
    It consists of three main parts:
    - backbone
    - rpn
    - object detection (roi_heads)
    - Scene graph parser model: IMP, MSDN, MOTIF, graph-rcnn, ect
    """

    def __init__(self, cfg):
        # GeneralizedRCNN.__init__(self, cfg)
        super(AttrRCNN, self).__init__(cfg)
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        feature_dim = self.backbone.out_channels

        if cfg.MODEL.ATTRIBUTE_ON:
            self.attribute = build_roi_attribute_head(cfg, feature_dim)
            if cfg.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                self.attribute.feature_extractor = self.roi_heads.box.feature_extractor

    def to(self, device, **kwargs):
        super(AttrRCNN, self).to(device, **kwargs)
        if self.cfg.MODEL.ATTRIBUTE_ON:
            self.attribute.to(device, **kwargs)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            We can assume that gt_boxlist contains two other fields:
                "relation_labels": list of [subj_id, obj_id, predicate_category]
                "pred_labels": n*n matrix with predicate_category (including BG) as values.

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        images = images.to(self.device)
        features = self.backbone(images.tensors)

        if targets:
            targets = [target.to(self.device)
                       for target in targets if target is not None]

        proposals, proposal_losses = self.rpn(images, features, targets)
        x, predictions, detector_losses = self.roi_heads(features,
                                                         proposals, targets)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            attribute_features = features
            # the attribute head reuse the features from the box head
            if (
                    self.training
                    and self.cfg.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                attribute_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x_attr, predictions, loss_attribute = self.attribute(
                attribute_features, predictions, targets
            )
            detector_losses.update(loss_attribute)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return predictions
