# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
"""
Relation head for predicting relationship between object pairs.
"""
import os.path as op

import numpy as np
import torch

from maskrcnn_benchmark.structures.bounding_box_pair import BoxPairList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, cat_boxlist_with_fields
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sparse_targets import FrequencyBias, _get_tensor_from_boxlist, _get_rel_inds

from .relpn.relpn import make_relation_proposal_network
from .baseline.baseline import build_baseline_model
from .neural_motif.neuralmotif import build_neuralmotif_model
from .imp.imp import build_imp_model
from .msdn.msdn import build_msdn_model
from .grcnn.grcnn import build_grcnn_model
from .reldn.reldn import build_reldn_model


class ROIRelationHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg
        self.force_relations = cfg.MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        if cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_baseline":
            self.rel_predictor = build_baseline_model(cfg, in_channels)
        elif cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_neuralmotif":
            self.rel_predictor = build_neuralmotif_model(cfg, in_channels)
        elif cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_imp":
            self.rel_predictor = build_imp_model(cfg, in_channels)
        elif cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_msdn":
            self.rel_predictor = build_msdn_model(cfg, in_channels)
        elif cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_grcnn":
            self.rel_predictor = build_grcnn_model(cfg, in_channels)
        elif cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_reldn":
            self.rel_predictor = build_reldn_model(cfg, in_channels)

        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_RELPN:
            self.relpn = make_relation_proposal_network(cfg)

        self.neural_motif_flag = cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM == "sg_neuralmotif"
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.USE_BIAS

        self.freq_dist = None
        if self.cfg.MODEL.USE_FREQ_PRIOR or self.use_bias:
            print("Using frequency bias: ", cfg.MODEL.FREQ_PRIOR)
            self.freq_dist_file = op.join(cfg.DATA_DIR, cfg.MODEL.FREQ_PRIOR)
            self.freq_dist = torch.from_numpy(np.load(self.freq_dist_file)).float()
            if self.cfg.MODEL.USE_FREQ_PRIOR:
                # never predict __no_relation__ for frequency prior
                self.freq_dist[:, :, 0] = 0
                # we use probability directly
                self.freq_bias = FrequencyBias(self.freq_dist)
            else:
                self.freq_dist = torch.log(self.freq_dist + 1e-3)
                self.freq_bias = FrequencyBias(self.freq_dist)
    
    def to(self, device, **kwargs):
        super(ROIRelationHead, self).to(device, **kwargs)
        if self.cfg.MODEL.USE_FREQ_PRIOR or self.use_bias:
            self.freq_bias.to(device, **kwargs)
        self.rel_predictor.to(device, **kwargs)

    def _get_proposal_pairs(self, proposals):
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals):
            box_subj = proposals_per_image.bbox
            box_obj = proposals_per_image.bbox

            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
            proposal_box_pairs = torch.cat(
                (box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(
                proposals_per_image.bbox.device)
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(
                proposals_per_image.bbox.device)
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

            label_subj = proposals_per_image.get_field('labels')[idx_subj]
            label_obj = proposals_per_image.get_field('labels')[idx_obj]
            proposal_label_pairs = torch.cat(
                (label_subj.view(-1, 1), label_obj.view(-1, 1)), 1)

            keep_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero(as_tuple=False).view(-1)

            # if we filter non overlap bounding boxes
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero(as_tuple=False).view(-1)]
            proposal_idx_pairs = proposal_idx_pairs[keep_idx]
            proposal_box_pairs = proposal_box_pairs[keep_idx]
            proposal_label_pairs = proposal_label_pairs[keep_idx]
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, proposals_per_image.size, proposals_per_image.mode)
            proposal_pairs_per_image.add_field("idx_pairs", proposal_idx_pairs)
            proposal_pairs_per_image.add_field("label_pairs", proposal_label_pairs)

            proposal_pairs.append(proposal_pairs_per_image)

        return proposal_pairs
    
    def _force_relation_pairs(self, targets):
        proposal_pairs = []
        for targets_per_image in targets:
            gt_triplets = targets_per_image.get_field('relation_labels')
            idx_subj = gt_triplets[:, 0]
            idx_obj = gt_triplets[:, 1]
            gt_box_labels = targets_per_image.get_field('labels')
            proposal_box_pairs = torch.cat((targets_per_image.bbox[idx_subj].view(-1, 4), targets_per_image.bbox[idx_obj].view(-1, 4)), 1)
            proposal_idx_pairs = gt_triplets[:, :2]
            proposal_label_pairs = torch.cat((gt_box_labels[idx_subj].view(-1, 1), gt_box_labels[idx_obj].view(-1, 1)), 1)
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, targets_per_image.size, targets_per_image.mode)
            proposal_pairs_per_image.add_field("idx_pairs", proposal_idx_pairs)
            proposal_pairs_per_image.add_field("label_pairs", proposal_label_pairs)
            proposal_pairs.append(proposal_pairs_per_image)
        return proposal_pairs

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # subsample proposals for Neural Motif
            # typically 64 per NM paper.
            if self.neural_motif_flag:
                proposals = self.loss_evaluator.sel_proposals(proposals,
                                                              self.cfg.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.NUM_OBJS)
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_RELPN:
                proposal_pairs, loss_relpn = self.relpn(proposals, targets)
            else:
                with torch.no_grad():
                    if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG:
                        proposal_pairs = self.loss_evaluator.contrastive_loss_sample(self.cfg, proposals, targets)
                    else:
                        # num relations sampled: ROI_HEADS.BATCH_SIZE_PER_IMAGE
                        # fraction of positive: ROI_HEADS.POSITIVE_FRACTION
                        proposal_pairs = self.loss_evaluator.subsample(proposals, targets)
            
            if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG or self.cfg.MODEL.ROI_RELATION_HEAD.CONCATENATE_PROPOSAL_GT:
                fields = ['box_features', 'labels', 'gt_labels']
                # if self.cfg.TEST.OUTPUT_SCORES_ALL:
                #     fields.append('scores_all')
                #     fields.append('boxes_all')
                if self.cfg.MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR:
                    fields += ['subj_box_features', 'obj_box_features']
                proposals = [cat_boxlist_with_fields([proposal_per_image, target], fields) for proposal_per_image, target in zip(proposals, targets)]
            
            if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG:
                self.loss_evaluator.contrastive_proposal_pair_transform(proposals, proposal_pairs)

        else:
            if self.force_relations:
                proposal_pairs = self._force_relation_pairs(targets)
            else:
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_RELPN:
                    proposal_pairs = self.relpn(proposals)
                else:
                    proposal_pairs = self._get_proposal_pairs(proposals)

        if self.cfg.MODEL.USE_FREQ_PRIOR:
            """
            if use frequency prior, we directly use the statistics
            """
            x = None
            obj_class_logits = None

            _, obj_labels, im_inds \
                = _get_tensor_from_boxlist(proposals, 'labels')
            _, proposal_idx_pairs, im_inds_pairs = _get_tensor_from_boxlist(
                proposal_pairs, 'idx_pairs')
            rel_inds = _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs, len(proposals))

            pred_class_logits = self.freq_bias.index_with_labels(
                torch.stack((
                    obj_labels[rel_inds[:, 0]],
                    obj_labels[rel_inds[:, 1]],
                ), 1))
        else:
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x, obj_class_logits, pred_class_logits, obj_class_preds, rel_inds \
                = self.rel_predictor(features, proposals, proposal_pairs)

            if self.use_bias:
                pred_class_logits = pred_class_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        obj_class_preds[rel_inds[:, 0]],
                        obj_class_preds[rel_inds[:, 1]],
                    ), 1))

        if not self.training:
            # # NOTE: if we have updated object class logits, then we need to update proposals as well!!!
            # if obj_class_logits is not None:
            #     boxes_per_image = [len(proposal) for proposal in proposals]
            #     obj_logits = obj_class_logits
            #     obj_scores, obj_labels = obj_class_logits[:, 1:].max(1)
            #     obj_labels = obj_labels + 1
            #     obj_logits = obj_logits.split(boxes_per_image, dim=0)
            #     obj_scores = obj_scores.split(boxes_per_image, dim=0)
            #     obj_labels = obj_labels.split(boxes_per_image, dim=0)
            #     for box_num, proposal, obj_logit, obj_score, obj_label in \
            #         zip(boxes_per_image, proposals, obj_logits, obj_scores, obj_labels):
            #         proposal.add_field("logits", obj_logit)
            #         proposal.add_field("scores", obj_score)
            #         proposal.add_field("labels", obj_label)
            #         if self.cfg.MODEL.ROI_RELATION_HEAD.UPDATE_BOX_REG and \
            #                 self.cfg.MODEL.ROI_RELATION_HEAD.MODE == "sgdet":
            #             box_inds = obj_label.data.new(box_num).long()
            #             twod_inds = torch.arange(0, box_num, out=box_inds) * self.num_classes + obj_label.data
            #             bboxes = proposal.get_field("boxes_all").view(-1, 4)[
            #                 twod_inds].view(box_num, 4)
            #             proposal.bbox = bboxes
            result = self.post_processor(pred_class_logits, proposal_pairs, x,
                                         use_freq_prior=self.cfg.MODEL.USE_FREQ_PRIOR)

            return x, result, {}

        loss_obj_classifier = torch.tensor(0, dtype=torch.float).to(pred_class_logits.device)
        if obj_class_logits is not None:
            loss_obj_classifier = self.loss_evaluator.obj_classification_loss(proposals, [obj_class_logits])

        if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG:
            # cross entropy loss
            loss_pred_classifier = self.loss_evaluator.cross_entropy_losses([pred_class_logits])

            # contrastive loss
            if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_LOSS:
                loss_contrastive_sbj, loss_contrastive_obj = self.loss_evaluator.reldn_contrastive_losses(
                    self.cfg, [pred_class_logits])
                loss_contrastive_sbj = loss_contrastive_sbj * self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_WEIGHT
                loss_contrastive_obj = loss_contrastive_obj * self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_WEIGHT
            if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS:
                loss_so_contrastive_sbj, loss_so_contrastive_obj = self.loss_evaluator.reldn_so_contrastive_losses(
                    self.cfg, [pred_class_logits])
                loss_so_contrastive_sbj = loss_so_contrastive_sbj * self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_WEIGHT
                loss_so_contrastive_obj = loss_so_contrastive_obj * self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_WEIGHT
            if self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
                loss_p_contrastive_sbj, loss_p_contrastive_obj = self.loss_evaluator.reldn_p_contrastive_losses(
                    self.cfg, [pred_class_logits])
                loss_p_contrastive_sbj = loss_p_contrastive_sbj * self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_WEIGHT
                loss_p_contrastive_obj = loss_p_contrastive_obj * self.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_WEIGHT

            return (
                x,
                proposal_pairs,
                dict(loss_obj_classifier=loss_obj_classifier, loss_pred_classifier=loss_pred_classifier, \
                    loss_contrastive_sbj=loss_contrastive_sbj, loss_contrastive_obj=loss_contrastive_obj, \
                    loss_so_contrastive_sbj=loss_so_contrastive_sbj, loss_so_contrastive_obj=loss_so_contrastive_obj, \
                    loss_p_contrastive_sbj=loss_p_contrastive_sbj, loss_p_contrastive_obj=loss_p_contrastive_obj),
            )
        else:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_RELPN:
                loss_pred_classifier = self.relpn.pred_classification_loss([pred_class_logits])
                return (
                    x,
                    proposal_pairs,
                    dict(loss_obj_classifier=loss_obj_classifier,
                        loss_relpn=loss_relpn,
                        loss_pred_classifier=loss_pred_classifier),
                )
            else:
                loss_pred_classifier = self.loss_evaluator([pred_class_logits])
                return (
                    x,
                    proposal_pairs,
                    dict(loss_obj_classifier=loss_obj_classifier,
                        loss_pred_classifier=loss_pred_classifier),
                )


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
