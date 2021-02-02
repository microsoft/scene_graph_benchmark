# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import torch
import torch.nn.functional as F
from torch import nn


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        cfg
    ):
        """
        Arguments:
            cfg
        """
        super(PostProcessor, self).__init__()
        self.output_feature = cfg.TEST.OUTPUT_RELATION_FEATURE
        if self.output_feature:
            # needed to extract features when they have not been pooled yet
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, boxes, features, use_freq_prior=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
                :param features: predicate features
                :param x: logits
                :param boxes: proposal pairs
                :param use_freq_prior: whether the logit is from frequency
        """
        class_logits = x
        class_prob = class_logits if use_freq_prior \
            else F.softmax(class_logits, -1)
        if self.output_feature:
            # TODO: ideally, we should have some more general way to always
            #       extract pooled features
            if isinstance(features, tuple):
                features = features[-1]
            if len(features.shape) > 2:
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]

        num_classes = class_prob.shape[1]

        proposals = boxes
        class_prob = class_prob.split(boxes_per_image, dim=0)
        if self.output_feature:
            features = features.split(boxes_per_image, dim=0)
        else:
            features = [None]*len(boxes_per_image)

        results = []
        for prob, boxes_per_img, image_shape, feature in zip(
            class_prob, proposals, image_shapes, features
        ):
            boxes_per_img.add_field("scores", prob)
            if self.output_feature:
                boxes_per_img.add_field("pred_features", feature)
            results.append(boxes_per_img)
        return results


def make_roi_relation_post_processor(cfg):
    postprocessor = PostProcessor(
        cfg
    )
    return postprocessor
