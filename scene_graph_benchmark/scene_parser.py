# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
"""
Implements the Scene Parser framework
"""
import numpy as np
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from .relation_head.relation_head import build_roi_relation_head
from maskrcnn_benchmark.modeling.backbone import build_backbone
from .relation_head.roi_relation_box_feature_extractors import make_roi_relation_box_feature_extractor
from .attribute_head.attribute_head import build_roi_attribute_head


class SceneParserOutputs(object):
    """
    Structure that holds SceneParser output object predicitions and relation predictions,
    and provide .to function to be able to move all nececssary tensors
    between gpu and cpu. (Inspired from SCANEmbedding)
    """
    def __init__(self, predictions, prediction_pairs=None):
        self.predictions = predictions
        self.prediction_pairs = prediction_pairs

    def to(self, *args, **kwargs):
        cast_predictions = self.predictions.to(*args, *kwargs)
        if self.prediction_pairs is not None:
            cast_prediction_pairs = self.prediction_pairs.to(*args, *kwargs)
        else:
            cast_prediction_pairs = None
        return SceneParserOutputs(cast_predictions, cast_prediction_pairs)


SCENE_PAESER_DICT = ["sg_baseline", "sg_imp", "sg_msdn", "sg_grcnn", "sg_reldn", "sg_neuralmotif"]


class SceneParser(GeneralizedRCNN):
    """
    Main class for Generalized Relation R-CNN.
    It consists of three main parts:
    - backbone
    - rpn
    - object detection (roi_heads)
    - Scene graph parser model: IMP, MSDN, MOTIF, graph-rcnn, ect
    """

    def __init__(self, cfg):
        super(SceneParser, self).__init__(cfg)
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.detector_pre_calculated = self.cfg.MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED
        self.detector_force_boxes = self.cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES
        self.cfg_check()

        feature_dim = self.backbone.out_channels
        if not self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_CONV_BACKBONE:
            self.rel_backbone = build_backbone(cfg)
            feature_dim = self.rel_backbone.out_channels

        # TODO: add force_relations logic
        self.force_relations = cfg.MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS
        if cfg.MODEL.RELATION_ON and self.cfg.MODEL.ROI_RELATION_HEAD.ALGORITHM in SCENE_PAESER_DICT:
            self.relation_head = build_roi_relation_head(cfg, feature_dim)
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.attribute_head = build_roi_attribute_head(cfg, feature_dim)

        # self._freeze_components(self.cfg)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.rpn.parameters():
            p.requires_grad = False
        for p in self.roi_heads.parameters():
            p.requires_grad = False
        
        if not self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            if self.cfg.MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR:
                self.subj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, feature_dim)
                self.obj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, feature_dim)
            else:
                self.obj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, feature_dim)

    def cfg_check(self):
        if self.cfg.MODEL.ROI_RELATION_HEAD.MODE=='predcls':
            assert self.cfg.MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED==False and self.cfg.MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS==False
        if self.cfg.MODEL.ROI_RELATION_HEAD.MODE=='sgcls':
            assert self.cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES==True and self.cfg.MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED==False

    def to(self, device, **kwargs):
        super(SceneParser, self).to(device, **kwargs)
        if self.cfg.MODEL.RELATION_ON:
            self.relation_head.to(device, **kwargs)
        if self.cfg.MODEL.ATTRIBUTE_ON:
            self.attribute_head.to(device, **kwargs)
        # if self.detector_pre_calculated:
        #     self.backbone.to('cpu')
        #     self.rpn.to('cpu')
        #     self.roi_heads.to('cpu')

    def _post_processing_constrained(self, result_obj, result_pred):
        """
        Arguments:
            object_predictions, predicate_predictions

        Returns:
            sort the object-predicate triplets, and output the top
        """
        result_obj_new, result_pred_new = [], []
        assert len(result_obj) == len(result_pred), "object list must have equal number to predicate list"
        for result_obj_i, result_pred_i in zip(result_obj, result_pred):
            obj_scores = result_obj_i.get_field("scores")
            rel_inds = result_pred_i.get_field("idx_pairs")
            pred_scores = result_pred_i.get_field("scores")
            scores = torch.stack((
                obj_scores[rel_inds[:,0]],
                obj_scores[rel_inds[:,1]],
                pred_scores[:,1:].max(1)[0]
            ), 1).prod(1)
            scores_sorted, order = scores.sort(0, descending=True)
            result_pred_i = result_pred_i[order[:self.cfg.MODEL.ROI_RELATION_HEAD.TRIPLETS_PER_IMG]]
            result_obj_new.append(result_obj_i)

            result_pred_i.add_field('labels', result_pred_i.get_field("scores")[:, 1:].argmax(dim=1)) # not include background
            result_pred_i.add_field('scores_all', result_pred_i.get_field('scores'))
            result_pred_i.add_field('scores', scores[order[:self.cfg.MODEL.ROI_RELATION_HEAD.TRIPLETS_PER_IMG]])
            # filter out bad prediction
            inds = result_pred_i.get_field('scores') > self.cfg.MODEL.ROI_RELATION_HEAD.POSTPROCESS_SCORE_THRESH
            result_pred_i = result_pred_i[inds]

            result_pred_new.append(result_pred_i)
        return result_obj_new, result_pred_new

    def _post_processing_unconstrained(self, result_obj, result_pred):
        """
        Arguments:
            object_predictions, predicate_predictions

        Returns:
            sort the object-predicate triplets, and output the top
        """
        result_obj_new, result_pred_new = [], []
        assert len(result_obj) == len(result_pred), "object list must have equal number to predicate list"
        for result_obj_i, result_pred_i in zip(result_obj, result_pred):
            obj_scores = result_obj_i.get_field("scores").cpu().numpy()
            rel_inds = result_pred_i.get_field("idx_pairs").cpu().numpy()
            pred_scores = result_pred_i.get_field("scores").cpu().numpy()[:, 1:]
            
            det_labels_prd = np.argsort(-pred_scores, axis=1)
            det_scores_prd = -np.sort(-pred_scores, axis=1)

            det_scores_so = obj_scores[rel_inds[:,0]] * obj_scores[rel_inds[:,1]]
            det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :2]

            det_scores_inds = argsort_desc(det_scores_spo)[:self.cfg.MODEL.ROI_RELATION_HEAD.TRIPLETS_PER_IMG]

            result_labels = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]

            result_pred_i = result_pred_i[det_scores_inds[:, 0]]
            result_pred_i.add_field('labels', torch.from_numpy(result_labels))
            result_pred_i.add_field('scores_all', result_pred_i.get_field('scores'))
            result_pred_i.add_field('scores', torch.from_numpy(det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]))
            # filter out bad prediction
            inds = result_pred_i.get_field('scores') > self.cfg.MODEL.ROI_RELATION_HEAD.POSTPROCESS_SCORE_THRESH
            result_pred_i = result_pred_i[inds]
            
            result_obj_new.append(result_obj_i)
            result_pred_new.append(result_pred_i)
        return result_obj_new, result_pred_new

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
        if self.force_relations and targets is None:
            # note targets cannot be None but could have 0 box.
            raise ValueError("In force_relations setting, targets should be passed")
        # set the object detector to evaluation mode and run the object detection model
        self.backbone.eval()
        self.rpn.eval()
        self.roi_heads.eval()

        images = to_image_list(images)
        if targets:
            if self.detector_pre_calculated:
                predictions = [prediction.to(self.device) for (target, prediction) in targets if prediction is not None]
                targets = [target.to(self.device) for (target, prediction) in targets if target is not None]
            else:
                targets = [target.to(self.device)
                        for target in targets if target is not None]

        scene_parser_losses = {}

        if not self.detector_pre_calculated:
            features = self.backbone(images.tensors)

            proposals, proposal_losses = self.rpn(images, features, targets)

            if self.detector_force_boxes:
                proposals = [BoxList(target.bbox, target.size, target.mode) for target in targets]
                x, predictions, detector_losses = self.roi_heads(features, proposals, targets)
            else:
                x, predictions, detector_losses = self.roi_heads(features, proposals, targets)
            scene_parser_losses.update(detector_losses)
        else:
            proposal_losses = {}
            if targets is not None or len(targets) != 0:
                predictions = self.roi_heads['box'].loss_evaluator.prepare_labels(predictions, targets)
        
        if (self.force_relations or self.cfg.MODEL.ROI_RELATION_HEAD.MODE=='predcls') and not self.training:
            predictions = targets
            for pred in predictions:
                pred.add_field('scores', torch.tensor([1.0]*len(pred)).to(self.device))
                if self.cfg.TEST.OUTPUT_FEATURE:
                    gt_labels = pred.get_field('labels')
                    gt_pseudo_scores_all = torch.zeros(len(pred), self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES).to(gt_labels.device)
                    gt_pseudo_scores_all.scatter_(1, gt_labels.unsqueeze(0).view(-1, 1), 1)
                    pred.add_field('scores_all', gt_pseudo_scores_all)
                    gt_boxes = pred.bbox
                    gt_pseudo_boxes_all = gt_boxes.unsqueeze(1).repeat(1, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, 1)
                    pred.add_field('boxes_all', gt_pseudo_boxes_all)
            if self.cfg.TEST.OUTPUT_FEATURE:
                gt_features = self.roi_heads.box.feature_extractor(features, predictions)
                if gt_features.ndimension() == 4:
                    gt_features = torch.nn.functional.adaptive_avg_pool2d(gt_features, 1)
                    gt_features = gt_features.view(gt_features.size(0), -1)
                gt_boxes_per_image = [len(box) for box in predictions]
                assert sum(gt_boxes_per_image)==len(gt_features), "gt_boxes_per_image and len(gt_features) do not match!"
                gt_features = gt_features.split(gt_boxes_per_image, dim=0)
                for pred, gt_feature in zip(predictions, gt_features):
                    pred.add_field('box_features', gt_feature)

        if not self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_CONV_BACKBONE:
            features = self.rel_backbone(images.tensors)
        else:
            features = [feature.detach() for feature in features]

        # relation classification network
        # optimization: during training, if we share the feature extractor between
        # the box and the relation heads, then we can reuse the features already computed
        if not self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            obj_features = self.obj_feature_extractor(features, predictions, use_relu=False)
            if obj_features.ndimension() == 4:
                obj_features = torch.nn.functional.adaptive_avg_pool2d(obj_features, 1)
                obj_features = obj_features.view(obj_features.size(0), -1)
            boxes_per_image = [len(box) for box in predictions]
            obj_features = obj_features.split(boxes_per_image, dim=0)
            for prediction, obj_feature in zip(predictions, obj_features):
                prediction.add_field('box_features', obj_feature)

            if self.cfg.MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR:
                subj_features = self.subj_feature_extractor(features, predictions, use_relu=False)
                if subj_features.ndimension() == 4:
                    subj_features = torch.nn.functional.adaptive_avg_pool2d(subj_features, 1)
                    subj_features = subj_features.view(subj_features.size(0), -1)
                boxes_per_image = [len(box) for box in predictions]
                subj_features = subj_features.split(boxes_per_image, dim=0)
                for prediction, subj_feature, obj_feature in zip(predictions, subj_features, obj_features):
                    prediction.add_field('subj_box_features', subj_feature)
                    prediction.add_field('obj_box_features', obj_feature)

        if self.training:
            if not self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                gt_features = self.obj_feature_extractor(features, targets, use_relu=False)
            else:
                gt_features = self.roi_heads.box.feature_extractor(features, targets)
            if gt_features.ndimension() == 4:
                gt_features = torch.nn.functional.adaptive_avg_pool2d(gt_features, 1)
                gt_features = gt_features.view(gt_features.size(0), -1)
            gt_boxes_per_image = [len(box) for box in targets]
            assert sum(gt_boxes_per_image)==len(gt_features), "gt_boxes_per_image and len(gt_features) do not match!"
            gt_features = gt_features.split(gt_boxes_per_image, dim=0)
            for target, gt_feature in zip(targets, gt_features):
                target.add_field('box_features', gt_feature)
                target.add_field('gt_labels', target.get_field('labels'))
                # if self.cfg.TEST.OUTPUT_SCORES_ALL:
                #     gt_labels = target.get_field('labels')
                #     gt_pseudo_scores_all = torch.zeros(len(target), self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES).to(gt_labels.device)
                #     gt_pseudo_scores_all.scatter_(1, gt_labels.unsqueeze(0).view(-1, 1), 1)
                #     target.add_field('scores_all', gt_pseudo_scores_all)
                #     gt_boxes = target.bbox
                #     gt_pseudo_boxes_all = gt_boxes.unsqueeze(1).repeat(1, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, 1)
                #     target.add_field('boxes_all', gt_pseudo_boxes_all)
            
            if self.cfg.MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR:
                gt_subj_features = self.subj_feature_extractor(features, targets, use_relu=False)
                if gt_subj_features.ndimension() == 4:
                    gt_subj_features = torch.nn.functional.adaptive_avg_pool2d(gt_subj_features, 1)
                    gt_subj_features = gt_subj_features.view(gt_subj_features.size(0), -1)
                gt_boxes_per_image = [len(box) for box in targets]
                gt_subj_features = gt_subj_features.split(gt_boxes_per_image, dim=0)
                for target, gt_subj_feature, gt_feature in zip(targets, gt_subj_features, gt_features):
                    target.add_field('subj_box_features', gt_subj_feature)
                    target.add_field('obj_box_features', gt_feature)
                
        # if not self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_CONV_BACKBONE:
        #     features = self.rel_backbone(images.tensors)
        # else:
        #     features = [feature.detach() for feature in features]

        # The predictions_pred contains idx_pairs (M*2) and scores (M*Pred_Cat); see Jianwei's code
        # TODO: add force_relations logic
        # pdb.set_trace()
        x_pairs, prediction_pairs, relation_losses = self.relation_head(features, predictions, targets)
        # pdb.set_trace()
        scene_parser_losses.update(relation_losses)

        # attribute head
        if self.cfg.MODEL.ATTRIBUTE_ON:
            x_attr, predictions, attribute_losses = self.attribute_head(features, predictions, targets)

        if self.training:
            losses = {}
            losses.update(scene_parser_losses)
            losses.update(proposal_losses)
            if self.cfg.MODEL.ATTRIBUTE_ON:
                losses.update(attribute_losses)
            return losses

        # NOTE: if object scores are updated in rel_heads, we need to ensure detections are updated accordingly
        if self.cfg.MODEL.ROI_RELATION_HEAD.POSTPROCESS_METHOD == 'constrained':
            predictions, prediction_pairs = self._post_processing_constrained(predictions, prediction_pairs)
        else:
            predictions, prediction_pairs = self._post_processing_unconstrained(predictions, prediction_pairs)

        return [SceneParserOutputs(prediction, prediction_pair)
                for prediction, prediction_pair in zip(predictions, prediction_pairs)]


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))