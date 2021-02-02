# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import torch
import torch.nn as nn
import numpy as np


def boxes_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.minimum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.maximum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()


def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw,
                         targets_dh)).transpose()
    return targets


class SpatialFeature(nn.Module):
    def __init__(self, cfg, dim):
        super(SpatialFeature, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 64), nn.LeakyReLU(0.1))

    def _get_pair_feature(self, boxes1, boxes2):
        delta_1 = bbox_transform_inv(boxes1, boxes2)
        delta_2 = bbox_transform_inv(boxes2, boxes1)
        spt_feat = np.hstack((delta_1, delta_2[:, :2]))
        return spt_feat

    def _get_box_feature(self, boxes, width, height):
        f1 = boxes[:, 0] / width
        f2 = boxes[:, 1] / height
        f3 = boxes[:, 2] / width
        f4 = boxes[:, 3] / height
        f5 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) / (width * height)
        return np.vstack((f1, f2, f3, f4, f5)).transpose()

    def _get_spt_features(self, boxes1, boxes2, width, height):
        boxes_u = boxes_union(boxes1, boxes2)
        spt_feat_1 = self._get_box_feature(boxes1, width, height)
        spt_feat_2 = self._get_box_feature(boxes2, width, height)
        spt_feat_12 = self._get_pair_feature(boxes1, boxes2)
        spt_feat_1u = self._get_pair_feature(boxes1, boxes_u)
        spt_feat_u2 = self._get_pair_feature(boxes_u, boxes2)
        return np.hstack((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1, spt_feat_2))

    def forward(self, proposal_pairs):
        spt_feats = []
        for i, proposal_pair in enumerate(proposal_pairs):
            boxes_subj = proposal_pair.bbox[:, :4]
            boxes_obj = proposal_pair.bbox[:, 4:]
            spt_feat = self._get_spt_features(boxes_subj.cpu().numpy(), boxes_obj.cpu().numpy(), proposal_pair.size[0], proposal_pair.size[1])
            spt_feat = torch.from_numpy(spt_feat).to(boxes_subj.device)
            spt_feats.append(spt_feat)
        spt_feats = torch.cat(spt_feats, 0).float()
        spt_feats = self.model(spt_feats)
        return spt_feats

def build_spatial_feature(cfg, dim=0):
    return SpatialFeature(cfg, dim)
