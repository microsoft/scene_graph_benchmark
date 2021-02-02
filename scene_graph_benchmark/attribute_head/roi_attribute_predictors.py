# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import torch
from torch import nn
from torch.nn import functional as F

from .. import registry


@registry.ROI_ATTRIBUTE_PREDICTOR.register("AttributeRCNNPredictor")
class AttributeRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(AttributeRCNNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        cls_emd_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.CLS_EMD_DIM
        num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        attr_emd_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTR_EMD_DIM

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_embedding=nn.Embedding(num_classes, cls_emd_dim)
        self.fc_attr=nn.Linear(in_channels+cls_emd_dim, attr_emd_dim)
        self.attr_score=nn.Linear(attr_emd_dim, num_attributes) 

        nn.init.normal_(self.cls_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc_attr.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_attr.bias, 0)
        nn.init.normal_(self.attr_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, labels):
        x = self.avgpool(x)
        pool5_flat = x.view(x.size(0), -1)

        cls_embedding=self.cls_embedding(labels)
        concat_pool5=torch.cat([pool5_flat,cls_embedding],1)
        fc_attr=F.relu(self.fc_attr(concat_pool5))

        return self.attr_score(fc_attr), fc_attr


@registry.ROI_ATTRIBUTE_PREDICTOR.register("AttributeFPNPredictor")
class AttributeFPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(AttributeFPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        cls_emd_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.CLS_EMD_DIM
        num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        attr_emd_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTR_EMD_DIM

        self.cls_embedding=nn.Embedding(num_classes, cls_emd_dim)
        self.fc_attr=nn.Linear(in_channels+cls_emd_dim, attr_emd_dim)
        self.attr_score=nn.Linear(attr_emd_dim, num_attributes) 

        nn.init.normal_(self.cls_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc_attr.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_attr.bias, 0)
        nn.init.normal_(self.attr_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, labels):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        cls_embedding=self.cls_embedding(labels)
        concat_x=torch.cat([x,cls_embedding],1)
        fc_attr=F.relu(self.fc_attr(concat_x))

        return self.attr_score(fc_attr), fc_attr


def make_roi_attribute_predictor(cfg, in_channels):
    func = registry.ROI_ATTRIBUTE_PREDICTOR[cfg.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR]
    return func(cfg, in_channels)
