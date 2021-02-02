# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


# TODO check if want to return a single BoxList or a composite
# object
class AttributePostProcessor(nn.Module):
    """
    From the results of the CNN, post process the attributes
    by taking the attributes corresponding to the class with max
    probability (which are padded to fixed size) and return the 
    attributes in the mask field of the BoxList.
    """

    def __init__(self, cfg):
        super(AttributePostProcessor, self).__init__()
        self.max_num_attr = cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_NUM_ATTR_PER_IMG
        self.max_num_attr_per_obj = cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_NUM_ATTR_PER_OBJ
        self.attr_thresh = cfg.MODEL.ROI_ATTRIBUTE_HEAD.POSTPROCESS_ATTRIBUTES_THRESHOLD
        self.output_feature = cfg.TEST.OUTPUT_ATTRIBUTE_FEATURE

    def forward(self, x, boxes, features):
        """
        Arguments:
            x (Tensor): the attribute logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image
            features (Tensor) : attribute features

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field attribute
        """
        boxes_per_image = [len(box) for box in boxes]
        attribute_probs = F.softmax(x, -1)
        num_classes = attribute_probs.shape[1]

        attribute_probs = attribute_probs.split(boxes_per_image, dim=0)
        features = features.split(boxes_per_image, dim=0)

        results = []
        for box, prob, feature in zip(boxes, attribute_probs, features):
            # copy the current boxes
            boxlist = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                boxlist.add_field(field, box.get_field(field))
            if self.output_feature:
                boxlist.add_field('attr_feature', feature)
            # filter out low probability and redundent boxes
            boxlist = self.filter_results(boxlist, prob, feature, num_classes)
            results.append(boxlist)

        return results

    def filter_results(self, boxlist, prob, feature, num_classes):
        """Returns feature detection results by thresholding on scores.
        """
        boxes = boxlist.bbox.reshape(-1, 4)
        scores = prob.reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities
        # Skip j = 0, because it's the background class
        scores[:, 0] = 0.0
        # filter by the attr_thresh
        inds_all = scores>self.attr_thresh
        number_of_detections = inds_all.sum().item()
        scores_all = scores
        scores[~inds_all] = 0.0
        # filter by max_num_attr_per_img
        if number_of_detections > self.max_num_attr > 0:
            attr_thresh, _ = torch.kthvalue(
                scores.flatten().cpu(), number_of_detections - self.max_num_attr + 1
            )
            scores[scores<attr_thresh.item()] = 0.0

        attr_scores, attr_labels = torch.topk(scores, self.max_num_attr_per_obj, dim=1)
        boxlist.add_field('attr_labels', attr_labels)
        boxlist.add_field('attr_scores', attr_scores)
        return boxlist


def make_roi_attribute_post_processor(cfg):
    attribute_post_processor = AttributePostProcessor(cfg)
    return attribute_post_processor
