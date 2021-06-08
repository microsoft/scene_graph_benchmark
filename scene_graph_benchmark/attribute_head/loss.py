# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

class AttributeRCNNLossComputation(object):
    def __init__(self, cfg):
        """
        """
        self.loss_weight = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT
        self.proposal_matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
                        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
                        allow_low_quality_matches=False,)

    def __call__(self, proposals, attribute_logits, targets=None):
        """
        Arguments:
            proposals (list[BoxList]): already contain gt_attributes
            attribute_logits (Tensor)

        Return:
            attribute_loss (Tensor): scalar tensor containing the loss
        """
        attributes = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")
            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS

            attributes_per_image = matched_targets.get_field("attributes")
            attributes_per_image = attributes_per_image.to(dtype=torch.int64)
            # Label background (below the low threshold)
            # attribute 0 is ignored in the loss
            attributes_per_image[bg_inds,:] = 0
            # Label ignore proposals (between low and high thresholds)
            attributes_per_image[ignore_inds,:] = 0

            attributes.append(attributes_per_image)

        attributes = torch.cat(attributes, dim=0)
        
        # prepare attribute targets
        sim_attributes = attribute_logits.new(attribute_logits.size()).zero_()
        for i in range(len(attributes)):
            if len(torch.nonzero(attributes[i], as_tuple=False)) > 0:
                sim_attributes[i][attributes[i][torch.nonzero(attributes[i], as_tuple=False)].long()] = 1.0 / len(
                    torch.nonzero(attributes[i], as_tuple=False))
        # TODO: do we need to ignore the all zero vector?
        attribute_loss = self.cross_entropy(attribute_logits, sim_attributes, loss_type="softmax")

        return self.loss_weight * attribute_loss

    def cross_entropy(self, pred, soft_targets, loss_type="softmax"):
        if loss_type == "sigmoid":
            return torch.mean(torch.sum(- soft_targets * F.logsigmoid(pred), 1))

        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))
    
    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "attributes"])
        if len(target) == 0:
            # take care of no positive labels during training
            dummy_box = torch.zeros((len(matched_idxs), 4), 
                    dtype=torch.float32, device=matched_idxs.device)
            matched_targets = BoxList(dummy_box, target.size, target.mode)
            matched_targets.add_field("labels", torch.zeros(len(matched_idxs),
                    dtype=torch.float32, device=matched_idxs.device))
        else:
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets


def make_roi_attribute_loss_evaluator(cfg):
    # no need to match any more because it is already done in box_head
    loss_evaluator = AttributeRCNNLossComputation(cfg)

    return loss_evaluator
