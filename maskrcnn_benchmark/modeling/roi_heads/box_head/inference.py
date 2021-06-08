# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.layers import nms as box_nms


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        cfg,
        box_coder=None,
    ):
        """
        Arguments:
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.cfg = cfg
        self.score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
        self.nms = cfg.MODEL.ROI_HEADS.NMS
        self.detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        self.min_detections_per_img = cfg.MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        self.bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
        self.output_feature = cfg.TEST.OUTPUT_FEATURE
        if self.output_feature:
            # needed to extract features when they have not been pooled yet
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.force_boxes = cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES
        self.ignore_box_regression = cfg.TEST.IGNORE_BOX_REGRESSION

        self.filter_method = self.filter_results
        if self.cfg.MODEL.ROI_HEADS.NMS_FILTER == 1:
            self.filter_method = self.filter_results_peter
        elif self.cfg.MODEL.ROI_HEADS.NMS_FILTER == 2:
            self.filter_method = self.filter_results_fast

    def forward(self, x, boxes, features):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image
            features (tensor): features that are used for prediction

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        if self.output_feature:
            # TODO: ideally, we should have some more general way to always
            #       extract pooled features
            if len(features.shape) > 2:
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)

        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        if self.ignore_box_regression or self.force_boxes:
            proposals = concat_boxes
        else:
            proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
        if (self.cls_agnostic_bbox_reg or self.ignore_box_regression) \
                and (self.cfg.MODEL.ROI_HEADS.NMS_FILTER != 2):
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        features = features.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape, feature in zip(
            class_prob, proposals, image_shapes, features
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if self.force_boxes:
                if len(boxlist) > 0:
                    # predict the most likely object in the box
                    # Skip j = 0, because it's the background class
                    scores, labels = torch.max(prob[:, 1:], dim=1)
                    boxlist.extra_fields['scores'] = scores
                    boxlist.add_field('labels', labels + 1)
                    if self.output_feature:
                        boxlist.add_field('box_features', feature)
                        boxlist.add_field('scores_all', prob)
                        boxlist.add_field('boxes_all',
                                          boxes_per_img.view(-1, 1, 4))
                else:
                    boxlist = self.prepare_empty_boxlist(boxlist)
            else:
                if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                    # to enforce minimum number of detections per image
                    # we will do a binary search on the confidence threshold
                    new_boxlist = self.filter_method(boxlist, num_classes, feature)

                    if self.cfg.MODEL.ROI_HEADS.NMS_FILTER == 2:
                        boxlist = new_boxlist
                    else:
                        initial_conf_thresh = self.score_thresh
                        decrease_num = 0
                        while new_boxlist.bbox.shape[0] < \
                                self.min_detections_per_img and decrease_num < 10:
                            self.score_thresh /= 2.0
                            print(("\nNumber of proposals {} is too small, "
                                    "retrying filter_results with score thresh"
                                    " = {}").format(new_boxlist.bbox.shape[0],
                                                    self.score_thresh))
                            new_boxlist = self.filter_method(boxlist, num_classes, feature)
                            decrease_num += 1
                        boxlist = new_boxlist
                        self.score_thresh = initial_conf_thresh
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def prepare_empty_boxlist(self, boxlist):
        device = boxlist.bbox.device
        # create an empty boxlist on cuda
        boxlist_empty = BoxList(torch.zeros((0, 4)).to(device),
                                boxlist.size, mode='xyxy')
        boxlist_empty.add_field("scores", torch.Tensor([]).to(device))
        boxlist_empty.add_field("labels", torch.full((0,), -1,
                                                     dtype=torch.int64,
                                                     device=device))
        if self.output_feature:
            boxlist_empty.add_field(
                "box_features",
                torch.full((0,), -1, dtype=torch.float32, device=device)
                )
            boxlist_empty.add_field(
                "scores_all", 
                torch.full((0,), -1, dtype=torch.float32, device=device)
                )
            boxlist_empty.add_field(
                "boxes_all", 
                torch.full((0,), -1, dtype=torch.float32, device=device)
                )
        return boxlist_empty

    def filter_results(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        boxlist_empty = self.prepare_empty_boxlist(boxlist)
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero(as_tuple=False).squeeze(1)

            if len(inds)>0:
                scores_j = scores[inds, j]
                boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                
                if self.output_feature:
                    feature_j = feature[inds]
                    boxlist_for_class.add_field("box_features", feature_j)
                    
                    scores_all = scores[inds]
                    boxlist_for_class.add_field("scores_all", scores_all)
                    boxlist_for_class.add_field("boxes_all",
                                                boxes[inds].view(-1, num_classes, 4))

                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
                )
                result.append(boxlist_for_class)
            else:
                result.append(boxlist_empty)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        return result

    def filter_results_peter(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        nms_mask = scores.clone()
        nms_mask.zero_()

        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        for j in range(1, num_classes):
            scores_j = scores[:, j]
            boxes_j = boxes[:, j * 4: (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("idxs",
                                        torch.arange(0, scores.shape[0]).long())
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, 0.3
            )
            nms_mask[:, j][boxlist_for_class.get_field("idxs")] = 1

        dists_all = nms_mask * scores

        # filter duplicate boxes
        scores_pre, labels_pre = dists_all.max(1)
        inds_pre = scores_pre.nonzero(as_tuple=False)
        assert inds_pre.dim() != 0
        inds_pre = inds_pre.squeeze(1)

        labels_pre = labels_pre[inds_pre]
        scores_pre = scores_pre[inds_pre]

        box_inds_pre = inds_pre * scores.shape[1] + labels_pre
        result = BoxList(boxlist.bbox.view(-1, 4)[box_inds_pre], boxlist.size,
                         mode="xyxy")
        result.add_field("labels", labels_pre)
        result.add_field("scores", scores_pre)
        if self.output_feature:
            features_pre = feature[inds_pre]
            result.add_field("box_features", features_pre)

            scores_all = scores[inds_pre]
            boxes_all = boxes[inds_pre]
            result.add_field("scores_all", scores_all)
            result.add_field("boxes_all", boxes_all.view(-1, num_classes, 4))

        vs, idx = torch.sort(scores_pre, dim=0, descending=True)
        keep_boxes = torch.nonzero(scores_pre >= self.score_thresh, as_tuple=True)[0]
        num_dets = len(keep_boxes)
        if num_dets < self.min_detections_per_img:
            keep_boxes = idx[:self.min_detections_per_img]
        elif num_dets > self.detections_per_img:
            keep_boxes = idx[:self.detections_per_img]
        else:
            keep_boxes = idx[:num_dets]

        result = result[keep_boxes]
        return result

    def filter_results_fast(self, boxlist, num_classes, feature=None):
        """ perform only one NMS for all classes.
        """
        assert boxlist.bbox.shape[1] == 4
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        # for each box, select max conf exclude background
        scores, labels = scores[:, 1:].max(1)
        labels += 1
        bbox = boxlist.bbox
        if not self.ignore_box_regression and not self.cls_agnostic_bbox_reg:
            bbox = bbox.reshape(-1, num_classes, 4).mean(1)

        boxlist.add_field("scores", scores)
        boxlist.add_field("labels", labels)
        boxlist.add_field("box_features", feature)

        # threshold by size and confidence
        # use a relatively low thresh to output enough boxes
        x1, y1, x2, y2 = bbox.split(1, dim=1)
        ws = (x2 - x1).squeeze(1)
        hs = (y2 - y1).squeeze(1)
        keep = (
            (ws >= 0) & (hs >= 0) & (scores > self.score_thresh * 0.01)
        ).nonzero(as_tuple=False).squeeze(1)
        del ws, hs

        # apply nms to the previous low-thresholded results
        nms_boxes = box_nms(bbox[keep], scores[keep], self.nms)
        nms_idx = keep[nms_boxes]  # indices that pass NMS and low-threshold
        nms_scores = scores[nms_idx]
        # sort above low-thresholded scores high to low
        _, idx = torch.sort(nms_scores, dim=0, descending=True)
        idx = nms_idx[idx]

        num_dets = (nms_scores >= self.score_thresh).long().sum()
        if not isinstance(num_dets, torch.Tensor):
            num_dets = torch.as_tensor(num_dets, device=scores.device)
        min_det = torch.stack([num_dets, torch.as_tensor(self.min_detections_per_img, device=scores.device)]).max()
        max_det = torch.stack([min_det, torch.as_tensor(self.detections_per_img, device=scores.device)]).min()

        keep_boxes = idx[:max_det]

        return boxlist[keep_boxes]


def make_roi_box_post_processor(cfg):
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    postprocessor = PostProcessor(
        cfg,
        box_coder,
    )
    return postprocessor
