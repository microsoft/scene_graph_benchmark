# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os.path as op
import numpy as np
import torch
import torch.nn as nn
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_predictors import make_roi_relation_predictor
from ..sparse_targets import FrequencyBias

from .spatial import build_spatial_feature

class RelDN(nn.Module):
    def __init__(self, cfg, in_channels, eps=1e-10):
        super(RelDN, self).__init__()
        self.cfg = cfg
        self.dim = 2048 if 'C4' in self.cfg.MODEL.BACKBONE.CONV_BODY \
            else self.cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)

        num_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.rel_embedding = nn.Sequential(
            nn.Linear(3 * self.dim, 3 * self.dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(3 * self.dim // 2, self.dim),
            nn.LeakyReLU(0.1),
        )

        self.rel_spatial_feat = build_spatial_feature(cfg, self.dim)

        self.rel_subj_predictor = make_roi_relation_predictor(cfg, self.dim)
        self.rel_obj_predictor = make_roi_relation_predictor(cfg, self.dim)
        self.rel_pred_predictor = make_roi_relation_predictor(cfg, self.dim)

        self.rel_spt_predictor = nn.Linear(64, num_classes)

        self.freq_dist_file = op.join(cfg.DATA_DIR, cfg.MODEL.FREQ_PRIOR)
        self.freq_dist = torch.from_numpy(np.load(self.freq_dist_file)).float()
        # self.pred_dist = torch.log(self.freq_dist + 1e-3) #10 * self.freq_dist
        # self.num_objs = self.pred_dist.shape[0]
        # self.pred_dist = torch.FloatTensor(self.pred_dist).view(-1, self.pred_dist.shape[2]).cuda()

        # self.num_objs = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1
        # self.pred_dist = self.freq_dist.cuda()

        self.pred_dist = torch.log(self.freq_dist + 1e-3)
        self.freq_bias = FrequencyBias(self.freq_dist, store_cpu=False)
        # for p in self.freq_bias.parameters():
        #     p.requires_grad = False
    
    def to(self, device, **kwargs):
        super(RelDN, self).to(device, **kwargs)
        self.freq_bias.to(device, **kwargs)

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field("idx_pairs").detach().clone()
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, 0)

        subj_pred_map = rel_inds.new(sum([len(proposal) for proposal in proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(sum([len(proposal) for proposal in proposals]), rel_inds.shape[0]).fill_(0).float().detach()

        subj_pred_map.scatter_(0, (rel_inds[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].contiguous().view(1, -1)), 1)

        return rel_inds, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        obj_class_logits = None
        rel_inds = []
        
        x_pred, _ = self.pred_feature_extractor(features, proposals, proposal_pairs, use_relu=False)
        if x_pred.ndimension() == 4:
            x_pred = self.avgpool(x_pred)
            x_pred = x_pred.view(x_pred.size(0), -1)

        '''compute spatial scores'''
        edge_spt_feats = self.rel_spatial_feat(proposal_pairs)
        rel_spt_class_logits = self.rel_spt_predictor(edge_spt_feats)

        x_obj = None
        
        sub_vert = []
        obj_vert = []
        rel_sem_class_logits = []
        offset = 0
        for img_id, (proposal_per_image, proposal_pairs_per_image) in \
                enumerate(zip(proposals, proposal_pairs)):
            rel_ind_i = proposal_pairs_per_image.get_field("idx_pairs").detach()
            rel_ind_i += offset
            offset += len(proposal_per_image)
            rel_inds.append(rel_ind_i)

            if self.cfg.MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR:
                sub_vert_per_image = proposal_per_image.get_field("subj_box_features")[rel_ind_i[:, 0]]
                obj_vert_per_image = proposal_per_image.get_field("obj_box_features")[rel_ind_i[:, 1]]
            else:
                x_obj_per_image = proposal_per_image.get_field("box_features")
                sub_vert_per_image = x_obj_per_image[rel_ind_i[:, 0]]
                obj_vert_per_image = x_obj_per_image[rel_ind_i[:, 1]]
            
            sub_vert.append(sub_vert_per_image)
            obj_vert.append(obj_vert_per_image)

            # get the semantic labels
            obj_labels = proposal_per_image.get_field("labels").detach()
            subj_vert_labels = obj_labels[rel_ind_i[:, 0]]
            obj_vert_labels = obj_labels[rel_ind_i[:, 1]]

            # class_logits_per_image = self.pred_dist[(subj_vert_labels-1) * self.num_objs + (obj_vert_labels-1)]
            class_logits_per_image = self.freq_bias.index_with_labels(torch.stack((subj_vert_labels, obj_vert_labels,), 1)) 
            rel_sem_class_logits.append(class_logits_per_image)

        sub_vert = torch.cat(sub_vert, 0)
        obj_vert = torch.cat(obj_vert, 0)
        rel_inds = torch.cat(rel_inds, 0)

        '''compute visual scores'''
        rel_subj_class_logits = self.rel_subj_predictor(sub_vert.unsqueeze(2).unsqueeze(3))
        rel_obj_class_logits = self.rel_obj_predictor(obj_vert.unsqueeze(2).unsqueeze(3))

        x_rel = torch.cat([sub_vert, x_pred, obj_vert], 1)
        x_rel = self.rel_embedding(x_rel)
        rel_pred_class_logits = self.rel_pred_predictor(x_rel.unsqueeze(2).unsqueeze(3))
        rel_vis_class_logits = rel_pred_class_logits + rel_subj_class_logits + rel_obj_class_logits

        '''compute semantic scores'''
        rel_sem_class_logits = torch.cat(rel_sem_class_logits, 0)
        
        '''sum up all scores'''
        rel_class_logits = rel_vis_class_logits + rel_sem_class_logits + rel_spt_class_logits

        if obj_class_logits is None:
            obj_class_labels = torch.cat([proposal.get_field("labels").detach() for proposal in proposals], 0)
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1

        return (x_obj, x_pred), obj_class_logits, rel_class_logits, obj_class_labels, rel_inds

def build_reldn_model(cfg, in_channels):
    return RelDN(cfg, in_channels)
