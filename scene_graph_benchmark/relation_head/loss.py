# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import numpy as np
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from .pair_matcher import PairMatcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box_pair import BoxPairList
from .balanced_positive_negative_pair_sampler import BalancedPositiveNegativePairSampler
from maskrcnn_benchmark.modeling.utils import cat
from .contrastive_loss_sample_pairs import add_rel_blobs, add_rel_proposals


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_pair_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False,
            use_matched_pairs_only=False,
            minimal_matched_pairs=10,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativePairSampler)
            box_coder (BoxCoder)
            use_matched_pairs_only: sample only among the pairs that have large iou with ground-truth pairs
            minimal_matched_pairs: if number of matched pairs is less than minimal_matched_pairs, disable use_matched_pairs_only
        """
        self.proposal_pair_matcher = proposal_matcher
        self.fg_bg_pair_sampler = fg_bg_pair_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.use_matched_pairs_only = use_matched_pairs_only
        self.minimal_matched_pairs = minimal_matched_pairs

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        temp = []
        target_box_pairs = []
        for i in range(match_quality_matrix.shape[0]):
            for j in range(match_quality_matrix.shape[0]):
                match_i = match_quality_matrix[i].view(-1, 1)
                match_j = match_quality_matrix[j].view(1, -1)
                match_ij = ((match_i + match_j) / 2)
                # rmeove duplicate index
                non_duplicate_idx = (torch.eye(match_ij.shape[0]).view(-1) == 0).nonzero(as_tuple=False).view(-1).to(match_ij.device)
                match_ij = match_ij.view(-1) # [::match_quality_matrix.shape[1]] = 0
                match_ij = match_ij[non_duplicate_idx]
                temp.append(match_ij)
                boxi = target.bbox[i]
                boxj = target.bbox[j]
                box_pair = torch.cat((boxi, boxj), 0)
                target_box_pairs.append(box_pair)

        match_pair_quality_matrix = torch.stack(temp, 0).view(len(temp), -1)
        target_box_pairs = torch.stack(target_box_pairs, 0)
        target_pair = BoxPairList(target_box_pairs, target.size, target.mode)
        target_pair.add_field("labels", target.get_field("pred_labels").view(-1))

        box_subj = proposal.bbox
        box_obj = proposal.bbox
        box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
        box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
        proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

        idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposal.bbox.device)
        idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposal.bbox.device)
        proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

        non_duplicate_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero(as_tuple=False)
        proposal_box_pairs = proposal_box_pairs[non_duplicate_idx.view(-1)]
        proposal_idx_pairs = proposal_idx_pairs[non_duplicate_idx.view(-1)]
        proposal_pairs = BoxPairList(proposal_box_pairs, proposal.size, proposal.mode)
        proposal_pairs.add_field("idx_pairs", proposal_idx_pairs)

        # matched_idxs = self.proposal_matcher(match_quality_matrix)
        matched_idxs = self.proposal_pair_matcher(match_pair_quality_matrix)

        # Fast RCNN only need "labels" field for selecting the targets
        # target = target.copy_with_fields("pred_labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        if self.use_matched_pairs_only and (matched_idxs >= 0).sum() > self.minimal_matched_pairs:
            # filter all matched_idxs < 0
            proposal_pairs = proposal_pairs[matched_idxs >= 0]
            matched_idxs = matched_idxs[matched_idxs >= 0]

        matched_targets = target_pair[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets, proposal_pairs

    # TODO: neural motif's relation assignment

    def prepare_targets(self, proposals, targets):
        labels = []
        proposal_pairs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, proposal_pairs_per_image = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            # regression_targets_per_image = self.box_coder.encode(
            #     matched_targets.bbox, proposals_per_image.bbox
            # )

            labels.append(labels_per_image)
            proposal_pairs.append(proposal_pairs_per_image)

            # regression_targets.append(regression_targets_per_image)

        return labels, proposal_pairs

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, proposal_pairs = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_pair_sampler(labels)

        proposal_pairs = list(proposal_pairs)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, proposal_pairs_per_image in zip(
                labels, proposal_pairs
        ):
            proposal_pairs_per_image.add_field("labels", labels_per_image)
            # proposals_per_image.add_field(
            #     "regression_targets", regression_targets_per_image
            # )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img, as_tuple=False).squeeze(1)
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
            proposal_pairs[img_idx] = proposal_pairs_per_image

        self._proposal_pairs = proposal_pairs
        return proposal_pairs

    # select top K based on classification scores
    def sel_proposals(self, proposals, num_objs=64):
        # select top ranked proposals from all ROIs.
        # for Neural Motif, typically 64 top proposals are returned
        for img_ind, props in enumerate(proposals):
            objness = props.extra_fields['scores']
            _, perm = torch.sort(objness, 0, descending=True)
            # select num_objs
            perm = perm[:num_objs]
            proposals[img_ind] = props[perm]

        return proposals

    def RelPN_GenerateProposalLabels(self, cfg, proposals, targets, proposal_box_pairs):
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        add_rel_proposals(proposals, targets, proposal_box_pairs)

        blobs = []
        for i in range(len(proposals)):
            output_blob_names = ['sbj_rois', 'obj_rois', 'rel_rois', 'fg_prd_labels_int32', 'all_prd_labels_int32', 'fg_size']
            if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPATIAL_FEAT:
                output_blob_names += ['spt_feat']
            if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FREQ_BIAS:
                output_blob_names += ['all_sbj_labels_int32']
                output_blob_names += ['all_obj_labels_int32']
            if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_LOSS or \
                            cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS or \
                            cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
                output_blob_names += ['binary_labels_sbj_pos_int32',
                                      'sbj_rois_sbj_pos', 'obj_rois_sbj_pos', 'rel_rois_sbj_pos',
                                      'spt_feat_sbj_pos',
                                      'sbj_labels_sbj_pos_int32', 'obj_labels_sbj_pos_int32', 'prd_labels_sbj_pos_int32',
                                      'sbj_labels_sbj_pos_fg_int32', 'obj_labels_sbj_pos_fg_int32',
                                      'inds_unique_sbj_pos',
                                      'inds_reverse_sbj_pos',
                                      'binary_labels_obj_pos_int32',
                                      'sbj_rois_obj_pos', 'obj_rois_obj_pos', 'rel_rois_obj_pos',
                                      'spt_feat_obj_pos',
                                      'sbj_labels_obj_pos_int32', 'obj_labels_obj_pos_int32', 'prd_labels_obj_pos_int32',
                                      'sbj_labels_obj_pos_fg_int32', 'obj_labels_obj_pos_fg_int32',
                                      'inds_unique_obj_pos',
                                      'inds_reverse_obj_pos']
            blob = {k: [] for k in output_blob_names}
            blobs.append(blob)
        
        add_rel_blobs(cfg, blobs, proposals, targets)

        return blobs

    def contrastive_loss_sample(self, cfg, proposals, targets):
        """Reldn Contrastive loss: generate a random sample of RoIs comprising foreground 
        and background examples.
        """
        proposal_box_pairs = []
        assert len(proposals) == len(targets) == 1

        box_subj = proposals[0].bbox
        box_obj = proposals[0].bbox

        box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
        box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
        proposal_box_pairs_per_image = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

        idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposals[0].bbox.device)
        idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposals[0].bbox.device)
        proposal_idx_pairs_per_image = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

        keep_idx = (proposal_idx_pairs_per_image[:, 0] != proposal_idx_pairs_per_image[:, 1]).nonzero(as_tuple=False).view(-1)

        # if we filter non overlap bounding boxes
        if cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
            ious = boxlist_iou(proposals[0], proposals[0]).view(-1)
            ious = ious[keep_idx]
            keep_idx = keep_idx[(ious > 0).nonzero(as_tuple=False).view(-1)]
        # proposal_idx_pairs_per_image = proposal_idx_pairs_per_image[keep_idx]
        proposal_box_pairs_per_image = proposal_box_pairs_per_image[keep_idx]
        proposal_box_pairs.append(proposal_box_pairs_per_image)

        # Add binary relationships
        blobs_out = self.RelPN_GenerateProposalLabels(cfg, proposals, targets, proposal_box_pairs)

        # get proposal_pairs
        proposal_pairs = []
        for i, (proposals_per_image, blob) in enumerate(zip(proposals, blobs_out)):
            # boxes and labels
            all_box_pairs = np.concatenate((blob['sbj_rois'], blob['obj_rois']), axis=1)
            all_label_pairs = np.concatenate((blob['all_sbj_labels_int32'].reshape((-1,1)), blob['all_obj_labels_int32'].reshape(-1,1)), axis=1)
            all_prd_label = blob['all_prd_labels_int32']

            sub_pos_box_pairs = np.concatenate((blob['sbj_rois_sbj_pos'], blob['obj_rois_sbj_pos']), axis=1)
            sub_pos_label_pairs = np.concatenate((blob['sbj_labels_sbj_pos_int32'].reshape((-1,1)), blob['obj_labels_sbj_pos_int32'].reshape(-1,1)), axis=1)
            sub_pos_prd_label = blob['prd_labels_sbj_pos_int32']

            obj_pos_box_pairs = np.concatenate((blob['sbj_rois_obj_pos'], blob['obj_rois_obj_pos']), axis=1)
            obj_pos_label_pairs = np.concatenate((blob['sbj_labels_obj_pos_int32'].reshape((-1,1)), blob['obj_labels_obj_pos_int32'].reshape(-1,1)), axis=1)
            obj_pos_prd_label = blob['prd_labels_obj_pos_int32']

            proposal_box_pairs = np.concatenate((all_box_pairs, sub_pos_box_pairs, obj_pos_box_pairs), axis=0)
            proposal_label_pairs = np.concatenate((all_label_pairs, sub_pos_label_pairs, obj_pos_label_pairs), axis=0)
            proposal_prd_label_pairs = np.concatenate((all_prd_label, sub_pos_prd_label, obj_pos_prd_label), axis=0)
            proposal_binary_all_pairs = np.zeros(proposal_box_pairs.shape[0], dtype=np.int32)
            proposal_binary_all_pairs[:all_box_pairs.shape[0]] = 1
            proposal_binary_label_sub_pairs = np.zeros(proposal_box_pairs.shape[0], dtype=np.int32)
            proposal_binary_label_sub_pairs[all_box_pairs.shape[0]:all_box_pairs.shape[0]+sub_pos_box_pairs.shape[0]] = 1
            proposal_binary_label_obj_pairs = np.zeros(proposal_box_pairs.shape[0], dtype=np.int32)
            proposal_binary_label_obj_pairs[all_box_pairs.shape[0]+sub_pos_box_pairs.shape[0]:] = 1

            # transform to cuda
            device = proposals_per_image.bbox.device

            proposal_box_pairs = torch.FloatTensor(proposal_box_pairs).to(device)
            proposal_label_pairs = torch.LongTensor(proposal_label_pairs).to(device)
            proposal_prd_label_pairs = torch.LongTensor(proposal_prd_label_pairs).to(device)
            proposal_binary_all_pairs = torch.LongTensor(proposal_binary_all_pairs).to(device)
            proposal_binary_label_sub_pairs = torch.LongTensor(proposal_binary_label_sub_pairs).to(device)
            proposal_binary_label_obj_pairs = torch.LongTensor(proposal_binary_label_obj_pairs).to(device)

            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, proposals_per_image.size, proposals_per_image.mode)
            proposal_pairs_per_image.add_field("label_pairs", proposal_label_pairs)
            proposal_pairs_per_image.add_field("prd_label_pairs", proposal_prd_label_pairs)
            proposal_pairs_per_image.add_field("binary_all_pairs", proposal_binary_all_pairs)
            proposal_pairs_per_image.add_field("binary_label_sub_pairs", proposal_binary_label_sub_pairs)
            proposal_pairs_per_image.add_field("binary_label_obj_pairs", proposal_binary_label_obj_pairs)

            # saved for contrastive losses (numpy)
            proposal_pairs_per_image.add_field("binary_labels_sbj_pos_int32", blob['binary_labels_sbj_pos_int32'])
            proposal_pairs_per_image.add_field("inds_unique_sbj_pos", blob['inds_unique_sbj_pos'])
            proposal_pairs_per_image.add_field("inds_reverse_sbj_pos", blob['inds_reverse_sbj_pos'])
            proposal_pairs_per_image.add_field("binary_labels_obj_pos_int32", blob['binary_labels_obj_pos_int32'])
            proposal_pairs_per_image.add_field("inds_unique_obj_pos", blob['inds_unique_obj_pos'])
            proposal_pairs_per_image.add_field("inds_reverse_obj_pos", blob['inds_reverse_obj_pos'])
            proposal_pairs_per_image.add_field("sbj_labels_sbj_pos_int32", blob['sbj_labels_sbj_pos_int32'])
            proposal_pairs_per_image.add_field("obj_labels_sbj_pos_int32", blob['obj_labels_sbj_pos_int32'])
            proposal_pairs_per_image.add_field("sbj_labels_obj_pos_int32", blob['sbj_labels_obj_pos_int32'])
            proposal_pairs_per_image.add_field("obj_labels_obj_pos_int32", blob['obj_labels_obj_pos_int32'])
            proposal_pairs_per_image.add_field("prd_labels_sbj_pos_int32", blob['prd_labels_sbj_pos_int32'])
            proposal_pairs_per_image.add_field("prd_labels_obj_pos_int32", blob['prd_labels_obj_pos_int32'])

            proposal_pairs.append(proposal_pairs_per_image)

        self._proposal_pairs = proposal_pairs
        return proposal_pairs
    
    def contrastive_proposal_pair_transform(self, proposals, proposal_pairs):
        for proposal_per_image, proposal_pairs_per_image in zip(proposals, proposal_pairs):
            device = proposal_per_image.bbox.device

            # add pseudo "idx_pairs" field for contrastive proposal pairs, so that they can be access similarly
            proposal_sub_obj = BoxList(
                    proposal_pairs_per_image.bbox.view(-1, 4),
                    proposal_pairs_per_image.size,
                    mode=proposal_pairs_per_image.mode
                )
            pair_obj_iou = boxlist_iou(proposal_sub_obj, proposal_per_image)
            vals, inds = torch.max(pair_obj_iou, 1)
            assert torch.min(vals) > 0.99, "Some sub/obj is not from proposals/targets!"
            
            rel_ind_i = torch.cat((inds[0::2].view(-1, 1), inds[1::2].view(-1, 1)), 1).to(device)
            proposal_pairs_per_image.add_field("idx_pairs", rel_ind_i)

    def __call__(self, class_logits):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])

        Returns:
            classification_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        # import pdb; pdb.set_trace()
        rel_fg_cnt = len(labels.nonzero(as_tuple=False))
        rel_bg_cnt = labels.shape[0] - rel_fg_cnt
        ce_weights = labels.new(class_logits.size(1)).fill_(1).float()
        ce_weights[0] = float(rel_fg_cnt) / (rel_bg_cnt + 1e-5)
        classification_loss = F.cross_entropy(class_logits, labels, weight=ce_weights)

        return classification_loss

    def obj_classification_loss(self, proposals, class_logits):
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device
        labels = cat([proposal.get_field("gt_labels") for proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        return classification_loss
    
    def cross_entropy_losses(self, class_logits):
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs

        prd_label_pairs = cat([proposal.get_field("prd_label_pairs") for proposal in proposals], dim=0)
        binary_all_pairs = cat([proposal.get_field("binary_all_pairs") for proposal in proposals], dim=0)

        keep_idx = binary_all_pairs==1
        prd_label_pairs = prd_label_pairs[keep_idx]
        class_logits = class_logits[keep_idx]
        classification_loss = F.cross_entropy(class_logits, prd_label_pairs)

        return classification_loss

    def reldn_contrastive_losses(self, cfg, class_logits, margin=0.2):
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs

        prd_label_pairs = cat([proposal.get_field("prd_label_pairs") for proposal in proposals], dim=0)
        binary_label_sub_pairs = cat([proposal.get_field("binary_label_sub_pairs") for proposal in proposals], dim=0)
        binary_label_obj_pairs = cat([proposal.get_field("binary_label_obj_pairs") for proposal in proposals], dim=0)
        prd_scores_sbj_pos = class_logits[binary_label_sub_pairs==1]
        prd_scores_obj_pos = class_logits[binary_label_obj_pairs==1]

        binary_labels_sbj_pos_int32 = np.concatenate([proposal.get_field("binary_labels_sbj_pos_int32") for proposal in proposals], axis=0)
        inds_unique_sbj_pos = np.concatenate([proposal.get_field("inds_unique_sbj_pos") for proposal in proposals], axis=0)
        inds_reverse_sbj_pos = np.concatenate([proposal.get_field("inds_reverse_sbj_pos") for proposal in proposals], axis=0)
        binary_labels_obj_pos_int32 = np.concatenate([proposal.get_field("binary_labels_obj_pos_int32") for proposal in proposals], axis=0)
        inds_unique_obj_pos = np.concatenate([proposal.get_field("inds_unique_obj_pos") for proposal in proposals], axis=0)
        inds_reverse_obj_pos = np.concatenate([proposal.get_field("inds_reverse_obj_pos") for proposal in proposals], axis=0)

        # sbj
        prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
        sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = self.split_pos_neg_spo_agnostic(
            prd_probs_sbj_pos, binary_labels_sbj_pos_int32, inds_unique_sbj_pos, inds_reverse_sbj_pos)
        sbj_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, \
            margin=cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_MARGIN)
        # obj
        prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
        obj_pair_pos_batch, obj_pair_neg_batch, obj_target = self.split_pos_neg_spo_agnostic(
            prd_probs_obj_pos, binary_labels_obj_pos_int32, inds_unique_obj_pos, inds_reverse_obj_pos)
        obj_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, \
            margin=cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_MARGIN)
        
        return sbj_contrastive_loss, obj_contrastive_loss

    def reldn_so_contrastive_losses(self, cfg, class_logits, margin=0.2):
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs

        prd_label_pairs = cat([proposal.get_field("prd_label_pairs") for proposal in proposals], dim=0)
        binary_label_sub_pairs = cat([proposal.get_field("binary_label_sub_pairs") for proposal in proposals], dim=0)
        binary_label_obj_pairs = cat([proposal.get_field("binary_label_obj_pairs") for proposal in proposals], dim=0)
        prd_scores_sbj_pos = class_logits[binary_label_sub_pairs==1]
        prd_scores_obj_pos = class_logits[binary_label_obj_pairs==1]

        binary_labels_sbj_pos_int32 = np.concatenate([proposal.get_field("binary_labels_sbj_pos_int32") for proposal in proposals], axis=0)
        inds_unique_sbj_pos = np.concatenate([proposal.get_field("inds_unique_sbj_pos") for proposal in proposals], axis=0)
        inds_reverse_sbj_pos = np.concatenate([proposal.get_field("inds_reverse_sbj_pos") for proposal in proposals], axis=0)
        binary_labels_obj_pos_int32 = np.concatenate([proposal.get_field("binary_labels_obj_pos_int32") for proposal in proposals], axis=0)
        inds_unique_obj_pos = np.concatenate([proposal.get_field("inds_unique_obj_pos") for proposal in proposals], axis=0)
        inds_reverse_obj_pos = np.concatenate([proposal.get_field("inds_reverse_obj_pos") for proposal in proposals], axis=0)
        sbj_labels_sbj_pos_int32 = np.concatenate([proposal.get_field("sbj_labels_sbj_pos_int32") for proposal in proposals], axis=0)
        obj_labels_sbj_pos_int32 = np.concatenate([proposal.get_field("obj_labels_sbj_pos_int32") for proposal in proposals], axis=0)
        sbj_labels_obj_pos_int32 = np.concatenate([proposal.get_field("sbj_labels_obj_pos_int32") for proposal in proposals], axis=0)
        obj_labels_obj_pos_int32 = np.concatenate([proposal.get_field("obj_labels_obj_pos_int32") for proposal in proposals], axis=0)

        # sbj
        prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
        sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = self.split_pos_neg_so_aware(
            cfg, prd_probs_sbj_pos,
            binary_labels_sbj_pos_int32, inds_unique_sbj_pos, inds_reverse_sbj_pos,
            sbj_labels_sbj_pos_int32, obj_labels_sbj_pos_int32, 's')
        sbj_so_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, \
            margin=cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_MARGIN)
        # obj
        prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
        obj_pair_pos_batch, obj_pair_neg_batch, obj_target = self.split_pos_neg_so_aware(
            cfg, prd_probs_obj_pos,
            binary_labels_obj_pos_int32, inds_unique_obj_pos, inds_reverse_obj_pos,
            sbj_labels_obj_pos_int32, obj_labels_obj_pos_int32, 'o')
        obj_so_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, \
            margin=cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_MARGIN)
        
        return sbj_so_contrastive_loss, obj_so_contrastive_loss

    def reldn_p_contrastive_losses(self, cfg, class_logits, margin=0.2):
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs

        prd_label_pairs = cat([proposal.get_field("prd_label_pairs") for proposal in proposals], dim=0)
        binary_label_sub_pairs = cat([proposal.get_field("binary_label_sub_pairs") for proposal in proposals], dim=0)
        binary_label_obj_pairs = cat([proposal.get_field("binary_label_obj_pairs") for proposal in proposals], dim=0)
        prd_scores_sbj_pos = class_logits[binary_label_sub_pairs==1]
        prd_scores_obj_pos = class_logits[binary_label_obj_pairs==1]

        binary_labels_sbj_pos_int32 = np.concatenate([proposal.get_field("binary_labels_sbj_pos_int32") for proposal in proposals], axis=0)
        inds_unique_sbj_pos = np.concatenate([proposal.get_field("inds_unique_sbj_pos") for proposal in proposals], axis=0)
        inds_reverse_sbj_pos = np.concatenate([proposal.get_field("inds_reverse_sbj_pos") for proposal in proposals], axis=0)
        binary_labels_obj_pos_int32 = np.concatenate([proposal.get_field("binary_labels_obj_pos_int32") for proposal in proposals], axis=0)
        inds_unique_obj_pos = np.concatenate([proposal.get_field("inds_unique_obj_pos") for proposal in proposals], axis=0)
        inds_reverse_obj_pos = np.concatenate([proposal.get_field("inds_reverse_obj_pos") for proposal in proposals], axis=0)
        prd_labels_sbj_pos_int32 = np.concatenate([proposal.get_field("prd_labels_sbj_pos_int32") for proposal in proposals], axis=0)
        prd_labels_obj_pos_int32 = np.concatenate([proposal.get_field("prd_labels_obj_pos_int32") for proposal in proposals], axis=0)

        # sbj
        prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
        prd_bias_probs_sbj_pos = None # not used
        # prd_bias_probs_sbj_pos = F.softmax(prd_bias_scores_sbj_pos, dim=1)
        sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = self.split_pos_neg_p_aware(
            cfg, prd_probs_sbj_pos,
            prd_bias_probs_sbj_pos,
            binary_labels_sbj_pos_int32, inds_unique_sbj_pos, inds_reverse_sbj_pos,
            prd_labels_sbj_pos_int32)
        sbj_p_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, \
            margin=cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_MARGIN)
        # obj
        prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
        prd_bias_probs_obj_pos = None # not used
        # prd_bias_probs_obj_pos = F.softmax(prd_bias_scores_obj_pos, dim=1)
        obj_pair_pos_batch, obj_pair_neg_batch, obj_target = self.split_pos_neg_p_aware(
            cfg, prd_probs_obj_pos,
            prd_bias_probs_obj_pos,
            binary_labels_obj_pos_int32, inds_unique_obj_pos, inds_reverse_obj_pos,
            prd_labels_obj_pos_int32)
        obj_p_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, \
            margin=cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_MARGIN)
        
        return sbj_p_contrastive_loss, obj_p_contrastive_loss

    def split_pos_neg_spo_agnostic(self, prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos):
        device_id = prd_probs.get_device()
        prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
        # loop over each group
        pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
        pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
        for i in range(inds_unique_pos.shape[0]):
            inds = np.where(inds_reverse_pos == i)[0]
            prd_pos_probs_i = prd_pos_probs[inds]
            binary_labels_pos_i = binary_labels_pos[inds]
            pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
            pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
            if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
                continue
            prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
            prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
            min_prd_pos_probs_i_pair_pos = torch.min(prd_pos_probs_i_pair_pos)
            max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)
            pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pair_pos.unsqueeze(0)))
            pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))

        target = torch.ones_like(pair_pos_batch).cuda(device_id)
            
        return pair_pos_batch, pair_neg_batch, target

    def split_pos_neg_so_aware(self, cfg, prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos, sbj_labels_pos, obj_labels_pos, s_or_o):
        device_id = prd_probs.get_device()
        prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
        # loop over each group
        pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
        pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
        for i in range(inds_unique_pos.shape[0]):
            inds = np.where(inds_reverse_pos == i)[0]
            prd_pos_probs_i = prd_pos_probs[inds]
            binary_labels_pos_i = binary_labels_pos[inds]
            sbj_labels_pos_i = sbj_labels_pos[inds]
            obj_labels_pos_i = obj_labels_pos[inds]
            pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
            pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
            if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
                continue
            prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
            prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
            sbj_labels_i_pair_pos = sbj_labels_pos_i[pair_pos_inds]
            obj_labels_i_pair_pos = obj_labels_pos_i[pair_pos_inds]
            sbj_labels_i_pair_neg = sbj_labels_pos_i[pair_neg_inds]
            obj_labels_i_pair_neg = obj_labels_pos_i[pair_neg_inds]
            max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
            if s_or_o == 's':
                # get all unique object labels
                unique_obj_labels, inds_unique_obj_labels, inds_reverse_obj_labels = np.unique(
                    obj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
                for j in range(inds_unique_obj_labels.shape[0]):
                    # get min pos
                    inds_j = np.where(inds_reverse_obj_labels == j)[0]
                    prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                    min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                    # get max neg
                    neg_j_inds = np.where(obj_labels_i_pair_neg == unique_obj_labels[j])[0]
                    if neg_j_inds.size == 0:
                        if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPO_AGNOSTIC_COMPENSATION:
                            pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                            pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                        continue
                    prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                    max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                    pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                    pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))
            else:
                # get all unique subject labels
                unique_sbj_labels, inds_unique_sbj_labels, inds_reverse_sbj_labels = np.unique(
                    sbj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
                for j in range(inds_unique_sbj_labels.shape[0]):
                    # get min pos
                    inds_j = np.where(inds_reverse_sbj_labels == j)[0]
                    prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                    min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                    # get max neg
                    neg_j_inds = np.where(sbj_labels_i_pair_neg == unique_sbj_labels[j])[0]
                    if neg_j_inds.size == 0:
                        if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPO_AGNOSTIC_COMPENSATION:
                            pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                            pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                        continue
                    prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                    max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                    pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                    pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

        target = torch.ones_like(pair_pos_batch).cuda(device_id)

        return pair_pos_batch, pair_neg_batch, target

    def split_pos_neg_p_aware(self, cfg, prd_probs, prd_bias_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos, prd_labels_pos):
        device_id = prd_probs.get_device()
        prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
        if len(prd_probs) == 0:
            prd_labels_det = torch.tensor([]).data.cpu().numpy()
        else:
            prd_labels_det = prd_probs[:, 1:].argmax(dim=1).data.cpu().numpy() + 1  # prd_probs is a torch.tensor, exlucding background
        # loop over each group
        pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
        pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
        for i in range(inds_unique_pos.shape[0]):
            inds = np.where(inds_reverse_pos == i)[0]
            prd_pos_probs_i = prd_pos_probs[inds]
            prd_labels_pos_i = prd_labels_pos[inds]
            prd_labels_det_i = prd_labels_det[inds]
            binary_labels_pos_i = binary_labels_pos[inds]
            pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
            pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
            if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
                continue
            prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
            prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
            prd_labels_i_pair_pos = prd_labels_pos_i[pair_pos_inds]
            prd_labels_i_pair_neg = prd_labels_det_i[pair_neg_inds]
            max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
            unique_prd_labels, inds_unique_prd_labels, inds_reverse_prd_labels = np.unique(
                prd_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_prd_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_prd_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(prd_labels_i_pair_neg == unique_prd_labels[j])[0]
                if neg_j_inds.size == 0:
                    if cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPO_AGNOSTIC_COMPENSATION:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

        target = torch.ones_like(pair_pos_batch).cuda(device_id)
            
        return pair_pos_batch, pair_neg_batch, target


def make_roi_relation_loss_evaluator(cfg):
    matcher = PairMatcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativePairSampler(
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
