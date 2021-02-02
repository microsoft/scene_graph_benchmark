# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import json
import os
from .tsv_dataset import TSVYamlDataset
from .utils.load_files import find_file_path_in_yaml
from .utils.label_loader import LabelLoader


class ODTSVDataset(TSVYamlDataset):
    """
    Generic TSV dataset format for Object Detection.
    """
    def __init__(self, yaml_file, extra_fields=(), transforms=None,
                 is_load_label=True, **kwargs):
        super(ODTSVDataset, self).__init__(yaml_file)

        self.transforms = transforms
        self.is_load_label = is_load_label
        self.attribute_on = kwargs['args'].MODEL.ATTRIBUTE_ON \
            if kwargs['args'] is not None else False

        if self.is_load_label:
            # construct maps
            jsondict_file = find_file_path_in_yaml(
                self.cfg.get("labelmap", None), self.root
            )
            jsondict = json.load(open(jsondict_file, 'r'))

            self.labelmap = {}
            self.class_to_ind = jsondict['label_to_idx']
            self.class_to_ind['__background__'] = 0
            self.ind_to_class = {v: k for k, v in self.class_to_ind.items()}
            self.labelmap['class_to_ind'] = self.class_to_ind

            if self.attribute_on:
                self.attribute_to_ind = jsondict['attribute_to_idx']
                self.attribute_to_ind['__no_attribute__'] = 0
                self.ind_to_attribute = {v:k for k, v in self.attribute_to_ind.items()}
                self.labelmap['attribute_to_ind'] = self.attribute_to_ind

            self.label_loader = LabelLoader(
                labelmap=self.labelmap,
                extra_fields=extra_fields,
            )

    def get_target_from_annotations(self, annotations, img_size, idx):
        if self.is_load_label:
            return self.label_loader(annotations['objects'], img_size)

    def apply_transforms(self, img, target=None):
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target