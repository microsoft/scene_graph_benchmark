# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import *
from .. import registry


registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)

registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "FPN2MLPFeatureExtractor", FPN2MLPFeatureExtractor
)

registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "FPNXconv1fcFeatureExtractor", FPNXconv1fcFeatureExtractor
)

registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "ViTHeadFeatureExtractor", ViTHeadFeatureExtractor
)


def make_roi_attribute_feature_extractor(cfg, in_channels):
    func = registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
