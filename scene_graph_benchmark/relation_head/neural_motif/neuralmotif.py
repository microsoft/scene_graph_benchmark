# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# Reimnplemetned by Pengchuan Zhang (penzhan@microsoft.com)
"""
Scene Graph Generation by Neural Motif
"""
import math
import json
import os.path as op

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_encoder import context_encoder
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_predictors import make_roi_relation_predictor
from ..sparse_targets import _get_tensor_from_boxlist


class NeuralMotif(nn.Module):
    def __init__(self, config, in_channels):
        super(NeuralMotif, self).__init__()

        # we need to build an obj level feat extractor in NM model
        # to be compatible with interface design in generalized_rcnn and relation head
        assert config.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR is False

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.HIDDEN_DIM
        self.use_online_obj_labels = config.MODEL.ROI_RELATION_HEAD.USE_ONLINE_OBJ_LABELS

        # feature extractor only for ResNet50Conv5ROIFeatureExtractor
        # TODO: extend to other feature extractos, like FPN2MLPRelationFeatureExtractor
        self.pred_feature_extractor = make_roi_relation_feature_extractor(config, in_channels)
        self.pred_feat_final_avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_objs = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.NUM_OBJS
        self.obj_feat_dim = self.pred_feature_extractor.out_channels

        # assign classes and relations labels
        # fn stands for filename
        obj_class_fn = op.join(config.DATA_DIR,
                               config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_CLASSES_FN)
        rel_class_fn = op.join(config.DATA_DIR,
                               config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.REL_CLASSES_FN)
        def sort_key_by_val(dic):
            sorted_dic = sorted(dic.items(), key=lambda kv: kv[1])
            return [kv[0] for kv in sorted_dic]
        with open(obj_class_fn, 'r') as f:
            # self.obj_classes = f.read().splitlines()
            self.class_to_ind = json.load(f)['label_to_idx']
            self.class_to_ind['__background__'] = 0
            self.obj_classes = sort_key_by_val(self.class_to_ind)
        with open(rel_class_fn, 'r') as f:
            # self.rel_classes = f.read().splitlines()
            self.relation_to_ind = json.load(f)['predicate_to_idx']
            self.relation_to_ind['__no_relation__'] = 0
            self.rel_classes = sort_key_by_val(self.relation_to_ind)
        self.num_classes = len(self.obj_classes)
        self.num_rels = len(self.rel_classes)

        # mode of Neural Motifs
        self.MODES = ('sgdet', 'sgcls', 'predcls')
        self.mode = config.MODEL.ROI_RELATION_HEAD.MODE
        # assert self.mode == 'sgdet', "Currently only sgdet is implemented..."

        self.use_tanh = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.USE_TANH

        self.context = context_encoder(config,
                                       self.obj_classes,
                                       self.rel_classes,
                                       self.obj_feat_dim)

        # post lstm fc, as per https://github.com/rowanz/neural-motifs/blob/master/lib/rel_model.py
        # Wh Wt combined
        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.
        if config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EDGE_LSTM_NUM_LAYERS > 0:
            # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
            self.post_lstm_fc = nn.Linear(self.hidden_dim,
                                          self.obj_feat_dim * 2)
            self.post_lstm_fc.weight.data.normal_(0, 10.0 * math.sqrt(
                1.0 / self.hidden_dim))
            self.post_lstm_fc.bias.data.zero_()
        else:
            self.post_emb = nn.Embedding(self.num_classes,
                                         self.obj_feat_dim * 2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        # out channel dim
        self.out_channels = self.obj_feat_dim

        # predictor
        self.predictor = make_roi_relation_predictor(config, self.out_channels)

    def _union_box_feats(self, x, proposals, proposal_pairs):
        # get union box's features
        x_union, rel_inds \
            = self.pred_feature_extractor(x, proposals, proposal_pairs)
        if x_union.ndimension() == 4:
            x_union = self.pred_feat_final_avgpool(x_union)
            x_union = x_union.view(x_union.size(0), -1)

        return x_union, rel_inds

    def forward(self, features, proposals, proposal_pairs):
        """
        Arguments:
            features (Tensor): features from rois.
                dim: B*rois_per_im, feat_dim
            proposals (list[BoxList]): detection results from detector
            proposal_pairs (list[BoxPairList]) : subsampled relation pairs

        Returns:
            for M proposal pairs:
            x (Tensor): the result of the feature extractor
                dim: M by feat_dim
            proposals_pairs (list[BoxPairList]): pair of boxes
            that forms possible relationships
        """

        obj_feats = torch.cat([prop.get_field("box_features")
                               for prop in proposals], dim=0)

        # acquire tensor format per batch data
        # bboxes, cls_prob (N, k)
        # im_inds: (N,1), img ind for each roi in the batch
        obj_box_priors, obj_detector_prob_dists, im_inds \
            = _get_tensor_from_boxlist(proposals, 'scores_all')
        _, boxes_all, _ \
            = _get_tensor_from_boxlist(proposals, 'boxes_all')

        # obj_gt_labels: (N,). Not one hot vector.
        obj_gt_labels = None
        if self.training:
            _, obj_gt_labels, _ \
                = _get_tensor_from_boxlist(proposals, 'gt_labels')
        elif self.mode == 'predcls':
            _, obj_gt_labels, _ \
                = _get_tensor_from_boxlist(proposals, 'labels')

        # get index in the proposal pairs
        edge_visual_feats, rel_inds = self._union_box_feats(features, proposals, proposal_pairs)

        # call context encoder
        obj_prob_dists, obj_preds, edge_ctx \
            = self.context(
            obj_feats,
            obj_detector_prob_dists.detach(),
            im_inds,
            obj_gt_labels,
            obj_box_priors,
            boxes_all
        )

        # fc to give each contexted obj feature two copies
        # first: representation for sub; second: for obj
        # Wh and Wt, eqn 6
        if edge_ctx is None:
            edge_rep = self.post_emb(obj_preds)
        else:
            edge_rep = self.post_lstm_fc(edge_ctx)

        # Split into subject and object representations
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.obj_feat_dim)
        subj_rep = edge_rep[:, 0]
        obj_rep = edge_rep[:, 1]

        # eqn 6 in NM paper
        edge_prod_rep = subj_rep[rel_inds[:, 0]] * obj_rep[rel_inds[:, 1]]

        # limit vision not implemented. another paper/code inconsistency
        edge_prod_rep = edge_prod_rep * edge_visual_feats

        if self.use_tanh:
            edge_prod_rep = F.tanh(edge_prod_rep)

        # predicate prediction
        class_logits = self.predictor(edge_prod_rep)

        if not self.use_online_obj_labels:
            obj_preds = torch.cat([proposal.get_field("labels") for proposal in proposals], 0)

        return edge_prod_rep, obj_prob_dists, class_logits, obj_preds, rel_inds

def build_neuralmotif_model(cfg, in_channels):
    return NeuralMotif(cfg, in_channels)
