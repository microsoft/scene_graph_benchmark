# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import torch
from torch.nn import functional as F
import scipy
from scipy import sparse
import numpy as np
import numpy.random as npr

def add_rel_blobs(cfg, blobs, proposals, targets):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for (blob, proposals_per_image, targets_per_image) in zip(blobs, proposals, targets):
        frcn_blobs = _sample_pairs(cfg, proposals_per_image, targets_per_image)
        for k, v in frcn_blobs.items():
            blob[k].append(v)
        # Concat the training blob lists into tensors
        for k, v in blob.items():
            if isinstance(v, list) and len(v) > 0:
                blob[k] = np.concatenate(v)
               
        # # ignore FPN setting 
        # if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        #     _add_rel_multilevel_rois(blobs)

    return True


def rois_union(rois1, rois2):
    xmin = np.minimum(rois1[:, 0], rois2[:, 0])
    ymin = np.minimum(rois1[:, 1], rois2[:, 1])
    xmax = np.maximum(rois1[:, 2], rois2[:, 2])
    ymax = np.maximum(rois1[:, 3], rois2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()


def _sample_pairs(cfg, proposals_per_image, targets_per_image):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    fg_pairs_per_image = cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_REL_SIZE_PER_IM
    pairs_per_image = int(cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_REL_SIZE_PER_IM \
                          / cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_REL_FRACTION)  # need much more pairs since it's quadratic
    max_pair_overlaps = targets_per_image.get_field('max_pair_overlaps')

    gt_pair_inds = np.where(max_pair_overlaps > 1.0 - 1e-4)[0]
    fg_pair_inds = np.where((max_pair_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                            (max_pair_overlaps <= 1.0 - 1e-4))[0]
    
    fg_pairs_per_this_image = np.minimum(fg_pairs_per_image, gt_pair_inds.size + fg_pair_inds.size)
    # Sample foreground regions without replacement
    if fg_pair_inds.size > 0 and fg_pairs_per_this_image > gt_pair_inds.size:
        fg_pair_inds = npr.choice(
            fg_pair_inds, size=(fg_pairs_per_this_image - gt_pair_inds.size), replace=False)
    fg_pair_inds = np.append(fg_pair_inds, gt_pair_inds)

    # Label is the class each RoI has max overlap with
    fg_prd_labels = targets_per_image.get_field('max_prd_classes')[fg_pair_inds]
    blob_dict = dict(
        fg_prd_labels_int32=fg_prd_labels.astype(np.int32, copy=False))
    if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_BG:
        bg_pair_inds = np.where((max_pair_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_HI))[0]
        
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_pairs_per_this_image = pairs_per_image - fg_pairs_per_this_image
        bg_pairs_per_this_image = np.minimum(bg_pairs_per_this_image, bg_pair_inds.size)
        # Sample foreground regions without replacement
        if bg_pair_inds.size > 0:
            bg_pair_inds = npr.choice(
                bg_pair_inds, size=bg_pairs_per_this_image, replace=False)
#         logger.info('{} : {}'.format(fg_pair_inds.size, bg_pair_inds.size))  
        keep_pair_inds = np.append(fg_pair_inds, bg_pair_inds)
        all_prd_labels = np.zeros(keep_pair_inds.size, dtype=np.int32)
        all_prd_labels[:fg_pair_inds.size] = fg_prd_labels + 1  # class should start from 1
    else:
        keep_pair_inds = fg_pair_inds
        all_prd_labels = fg_prd_labels
    blob_dict['all_prd_labels_int32'] = all_prd_labels.astype(np.int32, copy=False)
    blob_dict['fg_size'] = np.array([fg_pair_inds.size], dtype=np.int32)  # this is used to check if there is at least one fg to learn

    sampled_sbj_boxes = targets_per_image.get_field('sbj_boxes')[keep_pair_inds]
    sampled_obj_boxes = targets_per_image.get_field('obj_boxes')[keep_pair_inds]
    # Scale rois and format as (x1, y1, x2, y2)
    sampled_sbj_rois = sampled_sbj_boxes
    sampled_obj_rois = sampled_obj_boxes

    blob_dict['sbj_rois'] = sampled_sbj_rois
    blob_dict['obj_rois'] = sampled_obj_rois
    sampled_rel_rois = rois_union(sampled_sbj_rois, sampled_obj_rois)
    blob_dict['rel_rois'] = sampled_rel_rois
    if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPATIAL_FEAT:
        # ignore the spatial feat first
        pass
        # sampled_spt_feat = box_utils_rel.get_spt_features(
        #     sampled_sbj_boxes, sampled_obj_boxes, roidb['width'], roidb['height'])
        # blob_dict['spt_feat'] = sampled_spt_feat

    if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FREQ_BIAS:
        sbj_labels = targets_per_image.get_field('max_sbj_classes')[keep_pair_inds]
        obj_labels = targets_per_image.get_field('max_obj_classes')[keep_pair_inds]
        blob_dict['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False)
        blob_dict['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False)
    if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_LOSS or \
               cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS or \
               cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
        nodes_per_image = cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_SAMPLE_SIZE
        max_sbj_overlaps = targets_per_image.get_field('max_sbj_overlaps')
        max_obj_overlaps = targets_per_image.get_field('max_obj_overlaps')
        # sbj
        # Here a naturally existing assumption is, each positive sbj should have at least one positive obj
        sbj_pos_pair_pos_inds = np.where((max_pair_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH))[0]
        sbj_pos_obj_pos_pair_neg_inds = np.where((max_sbj_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_obj_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_HI))[0]
        sbj_pos_obj_neg_pair_neg_inds = np.where((max_sbj_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_obj_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_HI))[0]
        if sbj_pos_pair_pos_inds.size > 0:
            sbj_pos_pair_pos_inds = npr.choice(
                sbj_pos_pair_pos_inds,
                size=int(min(nodes_per_image, sbj_pos_pair_pos_inds.size)),
                replace=False)
        if sbj_pos_obj_pos_pair_neg_inds.size > 0:
            sbj_pos_obj_pos_pair_neg_inds = npr.choice(
                sbj_pos_obj_pos_pair_neg_inds,
                size=int(min(nodes_per_image, sbj_pos_obj_pos_pair_neg_inds.size)),
                replace=False)
        sbj_pos_pair_neg_inds = sbj_pos_obj_pos_pair_neg_inds
        if nodes_per_image - sbj_pos_obj_pos_pair_neg_inds.size > 0 and sbj_pos_obj_neg_pair_neg_inds.size > 0:
            sbj_pos_obj_neg_pair_neg_inds = npr.choice(
                sbj_pos_obj_neg_pair_neg_inds,
                size=int(min(nodes_per_image - sbj_pos_obj_pos_pair_neg_inds.size, sbj_pos_obj_neg_pair_neg_inds.size)),
                replace=False)
            sbj_pos_pair_neg_inds = np.append(sbj_pos_pair_neg_inds, sbj_pos_obj_neg_pair_neg_inds)
        sbj_pos_inds = np.append(sbj_pos_pair_pos_inds, sbj_pos_pair_neg_inds)
        binary_labels_sbj_pos = np.zeros(sbj_pos_inds.size, dtype=np.int32)
        binary_labels_sbj_pos[:sbj_pos_pair_pos_inds.size] = 1
        blob_dict['binary_labels_sbj_pos_int32'] = binary_labels_sbj_pos.astype(np.int32, copy=False)
        prd_pos_labels_sbj_pos = targets_per_image.get_field('max_prd_classes')[sbj_pos_pair_pos_inds]
        prd_labels_sbj_pos = np.zeros(sbj_pos_inds.size, dtype=np.int32)
        prd_labels_sbj_pos[:sbj_pos_pair_pos_inds.size] = prd_pos_labels_sbj_pos + 1
        blob_dict['prd_labels_sbj_pos_int32'] = prd_labels_sbj_pos.astype(np.int32, copy=False)
        sbj_labels_sbj_pos = targets_per_image.get_field('max_sbj_classes')[sbj_pos_inds] + 1
        # 1. set all obj labels > 0
        obj_labels_sbj_pos = targets_per_image.get_field('max_obj_classes')[sbj_pos_inds] + 1
        # 2. find those negative obj
        max_obj_overlaps_sbj_pos = targets_per_image.get_field('max_obj_overlaps')[sbj_pos_inds]
        obj_neg_inds_sbj_pos = np.where(max_obj_overlaps_sbj_pos < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH)[0]
        obj_labels_sbj_pos[obj_neg_inds_sbj_pos] = 0
        blob_dict['sbj_labels_sbj_pos_int32'] = sbj_labels_sbj_pos.astype(np.int32, copy=False)
        blob_dict['obj_labels_sbj_pos_int32'] = obj_labels_sbj_pos.astype(np.int32, copy=False)
        # this is for freq bias in RelDN
        blob_dict['sbj_labels_sbj_pos_fg_int32'] = targets_per_image.get_field('max_sbj_classes')[sbj_pos_inds].astype(np.int32, copy=False)
        blob_dict['obj_labels_sbj_pos_fg_int32'] = targets_per_image.get_field('max_obj_classes')[sbj_pos_inds].astype(np.int32, copy=False)
        
        sampled_sbj_boxes_sbj_pos = targets_per_image.get_field('sbj_boxes')[sbj_pos_inds]
        sampled_obj_boxes_sbj_pos = targets_per_image.get_field('obj_boxes')[sbj_pos_inds]
        # Scale rois and format as (x1, y1, x2, y2)
        sampled_sbj_rois_sbj_pos = sampled_sbj_boxes_sbj_pos
        sampled_obj_rois_sbj_pos = sampled_obj_boxes_sbj_pos
        blob_dict['sbj_rois_sbj_pos'] = sampled_sbj_rois_sbj_pos
        blob_dict['obj_rois_sbj_pos'] = sampled_obj_rois_sbj_pos
        sampled_rel_rois_sbj_pos = rois_union(sampled_sbj_rois_sbj_pos, sampled_obj_rois_sbj_pos)
        blob_dict['rel_rois_sbj_pos'] = sampled_rel_rois_sbj_pos
        _, inds_unique_sbj_pos, inds_reverse_sbj_pos = np.unique(
            sampled_sbj_rois_sbj_pos, return_index=True, return_inverse=True, axis=0)
        assert inds_reverse_sbj_pos.shape[0] == sampled_sbj_rois_sbj_pos.shape[0]
        blob_dict['inds_unique_sbj_pos'] = inds_unique_sbj_pos
        blob_dict['inds_reverse_sbj_pos'] = inds_reverse_sbj_pos
        if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPATIAL_FEAT:
            # ignore the spatial feat first
            pass
            # sampled_spt_feat_sbj_pos = box_utils_rel.get_spt_features(
            #     sampled_sbj_boxes_sbj_pos, sampled_obj_boxes_sbj_pos, roidb['width'], roidb['height'])
            # blob_dict['spt_feat_sbj_pos'] = sampled_spt_feat_sbj_pos
        # obj
        # Here a naturally existing assumption is, each positive obj should have at least one positive sbj
        obj_pos_pair_pos_inds = np.where((max_pair_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH))[0]
        obj_pos_sbj_pos_pair_neg_inds = np.where((max_obj_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_sbj_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_HI))[0]
        obj_pos_sbj_neg_pair_neg_inds = np.where((max_obj_overlaps >= cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_sbj_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_HI))[0]
        if obj_pos_pair_pos_inds.size > 0:
            obj_pos_pair_pos_inds = npr.choice(
                obj_pos_pair_pos_inds,
                size=int(min(nodes_per_image, obj_pos_pair_pos_inds.size)),
                replace=False)
        if obj_pos_sbj_pos_pair_neg_inds.size > 0:
            obj_pos_sbj_pos_pair_neg_inds = npr.choice(
                obj_pos_sbj_pos_pair_neg_inds,
                size=int(min(nodes_per_image, obj_pos_sbj_pos_pair_neg_inds.size)),
                replace=False)
        obj_pos_pair_neg_inds = obj_pos_sbj_pos_pair_neg_inds
        if nodes_per_image - obj_pos_sbj_pos_pair_neg_inds.size > 0 and obj_pos_sbj_neg_pair_neg_inds.size:
            obj_pos_sbj_neg_pair_neg_inds = npr.choice(
                obj_pos_sbj_neg_pair_neg_inds,
                size=int(min(nodes_per_image - obj_pos_sbj_pos_pair_neg_inds.size, obj_pos_sbj_neg_pair_neg_inds.size)),
                replace=False)
            obj_pos_pair_neg_inds = np.append(obj_pos_pair_neg_inds, obj_pos_sbj_neg_pair_neg_inds)
        obj_pos_inds = np.append(obj_pos_pair_pos_inds, obj_pos_pair_neg_inds)
        binary_labels_obj_pos = np.zeros(obj_pos_inds.size, dtype=np.int32)
        binary_labels_obj_pos[:obj_pos_pair_pos_inds.size] = 1
        blob_dict['binary_labels_obj_pos_int32'] = binary_labels_obj_pos.astype(np.int32, copy=False)
        prd_pos_labels_obj_pos = targets_per_image.get_field('max_prd_classes')[obj_pos_pair_pos_inds]
        prd_labels_obj_pos = np.zeros(obj_pos_inds.size, dtype=np.int32)
        prd_labels_obj_pos[:obj_pos_pair_pos_inds.size] = prd_pos_labels_obj_pos + 1
        blob_dict['prd_labels_obj_pos_int32'] = prd_labels_obj_pos.astype(np.int32, copy=False)
        obj_labels_obj_pos = targets_per_image.get_field('max_obj_classes')[obj_pos_inds] + 1
        # 1. set all sbj labels > 0
        sbj_labels_obj_pos = targets_per_image.get_field('max_sbj_classes')[obj_pos_inds] + 1
        # 2. find those negative sbj
        max_sbj_overlaps_obj_pos = targets_per_image.get_field('max_sbj_overlaps')[obj_pos_inds]
        sbj_neg_inds_obj_pos = np.where(max_sbj_overlaps_obj_pos < cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH)[0]
        sbj_labels_obj_pos[sbj_neg_inds_obj_pos] = 0
        blob_dict['sbj_labels_obj_pos_int32'] = sbj_labels_obj_pos.astype(np.int32, copy=False)
        blob_dict['obj_labels_obj_pos_int32'] = obj_labels_obj_pos.astype(np.int32, copy=False)
        # this is for freq bias in RelDN
        blob_dict['sbj_labels_obj_pos_fg_int32'] = targets_per_image.get_field('max_sbj_classes')[obj_pos_inds].astype(np.int32, copy=False)
        blob_dict['obj_labels_obj_pos_fg_int32'] = targets_per_image.get_field('max_obj_classes')[obj_pos_inds].astype(np.int32, copy=False)
        
        sampled_sbj_boxes_obj_pos = targets_per_image.get_field('sbj_boxes')[obj_pos_inds]
        sampled_obj_boxes_obj_pos = targets_per_image.get_field('obj_boxes')[obj_pos_inds]
        # Scale rois and format as (x1, y1, x2, y2)
        sampled_sbj_rois_obj_pos = sampled_sbj_boxes_obj_pos
        sampled_obj_rois_obj_pos = sampled_obj_boxes_obj_pos
        blob_dict['sbj_rois_obj_pos'] = sampled_sbj_rois_obj_pos
        blob_dict['obj_rois_obj_pos'] = sampled_obj_rois_obj_pos
        sampled_rel_rois_obj_pos = rois_union(sampled_sbj_rois_obj_pos, sampled_obj_rois_obj_pos)
        blob_dict['rel_rois_obj_pos'] = sampled_rel_rois_obj_pos
        _, inds_unique_obj_pos, inds_reverse_obj_pos = np.unique(
            sampled_obj_rois_obj_pos, return_index=True, return_inverse=True, axis=0)
        assert inds_reverse_obj_pos.shape[0] == sampled_obj_rois_obj_pos.shape[0]
        blob_dict['inds_unique_obj_pos'] = inds_unique_obj_pos
        blob_dict['inds_reverse_obj_pos'] = inds_reverse_obj_pos
        if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPATIAL_FEAT:
            # ignore the spatial feat first
            pass
            # sampled_spt_feat_obj_pos = box_utils_rel.get_spt_features(
            #     sampled_sbj_boxes_obj_pos, sampled_obj_boxes_obj_pos, roidb['width'], roidb['height'])
            # blob_dict['spt_feat_obj_pos'] = sampled_spt_feat_obj_pos

    return blob_dict


def add_rel_proposals(proposals, targets, proposal_box_pairs):
    sbj_box_list = []
    obj_box_list = []
    for i, (proposals_per_image, targets_per_image, proposal_box_pairs_per_image) \
                        in enumerate(zip(proposals, targets, proposal_box_pairs)):
        im_det_boxes = proposals_per_image.bbox.cpu().numpy()
        sbj_gt_boxes = targets_per_image.get_field('sbj_gt_boxes').cpu().numpy()
        obj_gt_boxes = targets_per_image.get_field('obj_gt_boxes').cpu().numpy()
        unique_sbj_gt_boxes = np.unique(sbj_gt_boxes, axis=0)
        unique_obj_gt_boxes = np.unique(obj_gt_boxes, axis=0)
        # sbj_gt w/ obj_det
        sbj_gt_boxes_paired_w_det = np.repeat(unique_sbj_gt_boxes, im_det_boxes.shape[0], axis=0)
        obj_det_boxes_paired_w_gt = np.tile(im_det_boxes, (unique_sbj_gt_boxes.shape[0], 1))
        # sbj_det w/ obj_gt
        sbj_det_boxes_paired_w_gt = np.repeat(im_det_boxes, unique_obj_gt_boxes.shape[0], axis=0)
        obj_gt_boxes_paired_w_det = np.tile(unique_obj_gt_boxes, (im_det_boxes.shape[0], 1))
        # sbj_gt w/ obj_gt
        sbj_gt_boxes_paired_w_gt = np.repeat(unique_sbj_gt_boxes, unique_obj_gt_boxes.shape[0], axis=0)
        obj_gt_boxes_paired_w_gt = np.tile(unique_obj_gt_boxes, (unique_sbj_gt_boxes.shape[0], 1))
        # now concatenate them all
        sbj_box_list.append(np.concatenate(
            (proposal_box_pairs_per_image.cpu().numpy()[:,:4], sbj_gt_boxes_paired_w_det, sbj_det_boxes_paired_w_gt, sbj_gt_boxes_paired_w_gt)))
        obj_box_list.append(np.concatenate(
            (proposal_box_pairs_per_image.cpu().numpy()[:,4:], obj_det_boxes_paired_w_gt, obj_gt_boxes_paired_w_det, obj_gt_boxes_paired_w_gt)))
    _merge_paired_boxes_into_roidb(proposals, targets, sbj_box_list, obj_box_list)
    _add_prd_class_assignments(proposals, targets)


def bbox_overlaps_np(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).reshape(1, K)

    anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) *
                    (anchors[:, 3] - anchors[:, 1] + 1)).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (np.minimum(boxes[:, :, 2], query_boxes[:, :, 2]) -
          np.maximum(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (np.minimum(boxes[:, :, 3], query_boxes[:, :, 3]) -
          np.maximum(boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def _merge_paired_boxes_into_roidb(proposals, targets, sbj_box_list, obj_box_list):
    assert len(sbj_box_list) == len(obj_box_list) == len(proposals) == len(targets) == 1
    for i, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
        sbj_boxes = sbj_box_list[i]
        obj_boxes = obj_box_list[i]
        assert sbj_boxes.shape[0] == obj_boxes.shape[0]
        num_pairs = sbj_boxes.shape[0]
        sbj_gt_overlaps = np.zeros(
            (num_pairs, targets_per_image.get_field('sbj_gt_overlaps').shape[1]),
            dtype=targets_per_image.get_field('sbj_gt_overlaps').cpu().numpy().dtype
        )
        obj_gt_overlaps = np.zeros(
            (num_pairs, targets_per_image.get_field('obj_gt_overlaps').shape[1]),
            dtype=targets_per_image.get_field('obj_gt_overlaps').cpu().numpy().dtype
        )
        prd_gt_overlaps = np.zeros(
            (num_pairs, targets_per_image.get_field('prd_gt_overlaps').shape[1]),
            dtype=targets_per_image.get_field('prd_gt_overlaps').cpu().numpy().dtype
        )
        pair_to_gt_ind_map = -np.ones(
            (num_pairs), dtype=targets_per_image.get_field('pair_to_gt_ind_map').cpu().numpy().dtype
        )
        
        pair_gt_inds = np.arange(targets_per_image.get_field('prd_gt_classes_minus_1').shape[0])
        if len(pair_gt_inds) > 0:
            sbj_gt_boxes = targets_per_image.get_field('sbj_gt_boxes').cpu().numpy()[pair_gt_inds, :]
            sbj_gt_classes_minus_1 = targets_per_image.get_field('sbj_gt_classes_minus_1').cpu().numpy()[pair_gt_inds]
            obj_gt_boxes = targets_per_image.get_field('obj_gt_boxes').cpu().numpy()[pair_gt_inds, :]
            obj_gt_classes_minus_1 = targets_per_image.get_field('obj_gt_classes_minus_1').cpu().numpy()[pair_gt_inds]
            prd_gt_classes_minus_1 = targets_per_image.get_field('prd_gt_classes_minus_1').cpu().numpy()[pair_gt_inds]
            sbj_to_gt_overlaps = bbox_overlaps_np(
                sbj_boxes.astype(dtype=np.float32, copy=False),
                sbj_gt_boxes.astype(dtype=np.float32, copy=False)
            )
            obj_to_gt_overlaps = bbox_overlaps_np(
                obj_boxes.astype(dtype=np.float32, copy=False),
                obj_gt_boxes.astype(dtype=np.float32, copy=False)
            )
            pair_to_gt_overlaps = np.minimum(sbj_to_gt_overlaps, obj_to_gt_overlaps)

            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            sbj_argmaxes = sbj_to_gt_overlaps.argmax(axis=1)
            sbj_maxes = sbj_to_gt_overlaps.max(axis=1)  # Amount of that overlap
            sbj_I = np.where(sbj_maxes >= 0)[0]  # Those boxes with non-zero overlap with gt boxes, get all items
            
            obj_argmaxes = obj_to_gt_overlaps.argmax(axis=1)
            obj_maxes = obj_to_gt_overlaps.max(axis=1)  # Amount of that overlap
            obj_I = np.where(obj_maxes >= 0)[0]  # Those boxes with non-zero overlap with gt boxes, get all items
            
            pair_argmaxes = pair_to_gt_overlaps.argmax(axis=1)
            pair_maxes = pair_to_gt_overlaps.max(axis=1)  # Amount of that overlap
            pair_I = np.where(pair_maxes >= 0)[0]  # Those boxes with non-zero overlap with gt boxes, get all items
            # Record max overlaps with the class of the appropriate gt box
            sbj_gt_overlaps[sbj_I, sbj_gt_classes_minus_1[sbj_argmaxes[sbj_I]]] = sbj_maxes[sbj_I]
            obj_gt_overlaps[obj_I, obj_gt_classes_minus_1[obj_argmaxes[obj_I]]] = obj_maxes[obj_I]
            prd_gt_overlaps[pair_I, prd_gt_classes_minus_1[pair_argmaxes[pair_I]]] = pair_maxes[pair_I]
            pair_to_gt_ind_map[pair_I] = pair_gt_inds[pair_argmaxes[pair_I]]

        sbj_boxes = sbj_boxes.astype(targets_per_image.get_field('sbj_gt_boxes').cpu().numpy().dtype, copy=False)
        targets_per_image.add_field('sbj_boxes', sbj_boxes)
        targets_per_image.add_field('sbj_gt_overlaps', scipy.sparse.csr_matrix(sbj_gt_overlaps))

        obj_boxes = obj_boxes.astype(targets_per_image.get_field('obj_gt_boxes').cpu().numpy().dtype, copy=False)
        targets_per_image.add_field('obj_boxes', obj_boxes)
        targets_per_image.add_field('obj_gt_overlaps', scipy.sparse.csr_matrix(obj_gt_overlaps))

        prd_gt_classes_minus_1 = -np.ones((num_pairs), dtype=targets_per_image.get_field('prd_gt_classes_minus_1').cpu().numpy().dtype)
        targets_per_image.add_field('prd_gt_overlaps', scipy.sparse.csr_matrix(prd_gt_overlaps))
        pair_to_gt_ind_map = pair_to_gt_ind_map.astype(targets_per_image.get_field('pair_to_gt_ind_map').cpu().numpy().dtype, copy=False)
        targets_per_image.add_field('pair_to_gt_ind_map', pair_to_gt_ind_map)


def _add_prd_class_assignments(proposals, targets):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for (proposals_per_image, targets_per_image) in zip(proposals, targets):
        sbj_gt_overlaps = targets_per_image.get_field('sbj_gt_overlaps').toarray()
        max_sbj_overlaps = sbj_gt_overlaps.max(axis=1)
        max_sbj_classes = sbj_gt_overlaps.argmax(axis=1)
        targets_per_image.add_field('max_sbj_classes', max_sbj_classes)
        targets_per_image.add_field('max_sbj_overlaps', max_sbj_overlaps)

        obj_gt_overlaps = targets_per_image.get_field('obj_gt_overlaps').toarray()
        max_obj_overlaps = obj_gt_overlaps.max(axis=1)
        max_obj_classes = obj_gt_overlaps.argmax(axis=1)
        targets_per_image.add_field('max_obj_classes', max_obj_classes)
        targets_per_image.add_field('max_obj_overlaps', max_obj_overlaps)

        prd_gt_overlaps = targets_per_image.get_field('prd_gt_overlaps').toarray()
        max_pair_overlaps = prd_gt_overlaps.max(axis=1)
        max_prd_classes = prd_gt_overlaps.argmax(axis=1)
        targets_per_image.add_field('max_prd_classes', max_prd_classes)
        targets_per_image.add_field('max_pair_overlaps', max_pair_overlaps)

        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        # zero_inds = np.where(max_pair_overlaps == 0)[0]
        # assert all(max_prd_classes[zero_inds] == 0)
        # # if max overlap > 0, the class must be a fg class (not class 0)
        # nonzero_inds = np.where(max_pair_overlaps > 0)[0]
        # assert all(max_prd_classes[nonzero_inds] != 0)