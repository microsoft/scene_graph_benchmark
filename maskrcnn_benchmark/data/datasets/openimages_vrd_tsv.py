# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
from .relation_tsv import RelationTSVDataset


class OpenImagesVRDTSVDataset(RelationTSVDataset):
    def __init__(self, yaml_file, extra_fields=(), transforms=None,
            is_load_label=True, **kwargs):

        super(OpenImagesVRDTSVDataset, self).__init__(yaml_file, extra_fields, transforms, is_load_label, **kwargs)