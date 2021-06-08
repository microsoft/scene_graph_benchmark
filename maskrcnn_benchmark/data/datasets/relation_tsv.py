# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import json
import torch
import math
import numpy as np

from maskrcnn_benchmark.structures.tsv_file import TSVFile
from .tsv_dataset import TSVYamlDataset
from .utils.load_files import find_file_path_in_yaml
from .utils.label_loader import LabelLoader
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


def sort_key_by_val(dic):
    sorted_dic = sorted(dic.items(), key=lambda kv: kv[1])
    return [kv[0] for kv in sorted_dic]


class RelationTSVDataset(TSVYamlDataset):
    """
    Generic TSV dataset format for Object Detection.
    """
    def __init__(self, yaml_file, extra_fields=(), transforms=None,
                 is_load_label=True, **kwargs):
        super(RelationTSVDataset, self).__init__(yaml_file)

        self.transforms = transforms
        self.is_load_label = is_load_label
        self.relation_on = kwargs['args'].MODEL.RELATION_ON if kwargs['args'] is not None else False
        self.attribute_on = kwargs['args'].MODEL.ATTRIBUTE_ON if kwargs['args'] is not None else False
        self.detector_pre_calculated = kwargs['args'].MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED if kwargs['args'] is not None else False

        self.contrastive_loss_on = kwargs['args'].MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG if kwargs['args'] is not None else False
        
        # construct maps
        jsondict_file = find_file_path_in_yaml(self.cfg.get("labelmap", self.cfg.get("jsondict", None)), self.root) # previous version use jsondict
        jsondict = json.load(open(jsondict_file, 'r'))

        self.labelmap = {}

        self.class_to_ind = jsondict['label_to_idx']
        self.class_to_ind['__background__'] = 0
        self.ind_to_class = {v:k for k, v in self.class_to_ind.items()}
        self.labelmap['class_to_ind'] = self.class_to_ind
        self.classes = sort_key_by_val(self.class_to_ind)

        if self.attribute_on:
            self.attribute_to_ind = jsondict['attribute_to_idx']
            self.attribute_to_ind['__no_attribute__'] = 0
            self.ind_to_attribute = {v:k for k, v in self.attribute_to_ind.items()}
            self.labelmap['attribute_to_ind'] = self.attribute_to_ind
            self.attributes = sort_key_by_val(self.attribute_to_ind)

        if self.relation_on:
            self.relation_to_ind = jsondict['predicate_to_idx']
            self.relation_to_ind['__no_relation__'] = 0
            self.ind_to_relation = {v:k for k, v in self.relation_to_ind.items()}
            self.labelmap['relation_to_ind'] = self.relation_to_ind
            self.relations = sort_key_by_val(self.relation_to_ind)
        
        if self.is_load_label or self.detector_pre_calculated:
            self.label_loader = LabelLoader(
                labelmap=self.labelmap,
                extra_fields=extra_fields,
            )
        
        # load pre-calculated object detection bounding boxes and features
        self.predictedbbox_conf_threshold = 0.0
        self.detections_per_img = kwargs['args'].MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        if self.detector_pre_calculated:
            self.predictedbbox_conf_threshold = kwargs['args'].MODEL.ROI_RELATION_HEAD.DETECTOR_BOX_THRESHOLD
            predictedbbox_file = find_file_path_in_yaml(self.cfg.get('predictedbbox', None), self.root)
            self.predictedbbox_tsv = None if predictedbbox_file is None else TSVFile(predictedbbox_file)

    def get_target_from_annotations(self, annotations, img_size):
        if self.is_load_label:
            # do not remove empty box to make sure all boxes are loaded in order.
            target = self.label_loader(annotations['objects'], img_size, remove_empty=False)
            # load relations
            if self.relation_on:
                target = self.label_loader.relation_loader(annotations["relations"], target)
            return target

    def apply_transforms(self, img, target=None, pre_calculate_boxlist=None):
        if self.transforms is not None:
            if pre_calculate_boxlist is not None:
                temp_box_list = cat_boxlist([target.copy_with_fields(['labels']), pre_calculate_boxlist.copy_with_fields(['labels'])]) if target is not None else pre_calculate_boxlist
            else:
                temp_box_list = target
            img, temp_box_list = self.transforms(img, temp_box_list)
            if pre_calculate_boxlist is not None:
                if target is not None:
                    bboxes = temp_box_list.bbox.split([len(target), len(pre_calculate_boxlist)], dim=0)
                    target.size = temp_box_list.size
                    target.bbox = bboxes[0]
                    pre_calculate_boxlist.size = temp_box_list.size
                    pre_calculate_boxlist.bbox = bboxes[1]
                else:
                    pre_calculate_boxlist.size = temp_box_list.size
                    pre_calculate_boxlist.bbox = temp_box_list.bbox
            else:
                target = temp_box_list
        return img, target, pre_calculate_boxlist

    def load_detection_result(self, idx, img_size, remove_empty=True, conf_threshold=0.0, detections_per_img=100):
        line_no = self.get_line_no(idx)
        if self.predictedbbox_tsv is not None:
            row = self.predictedbbox_tsv.seek(line_no)
            assert row[0] == self.get_img_key(idx)
            predicted_bboxes = json.loads(row[1])
            if 'objects' in predicted_bboxes:
                predicted_bboxes = predicted_bboxes['objects']

            # ['class', 'conf', 'scores_all', 'boxes_all', 'feature']:
            target = self.label_loader(predicted_bboxes, img_size, remove_empty=remove_empty, load_fields=['class', 'conf'])

            # filter according to the confidence
            selected_idxes = target.get_field('scores')>=conf_threshold
            while torch.sum(selected_idxes) < 2:
                conf_threshold = conf_threshold / 2
                selected_idxes = target.get_field('scores')>=conf_threshold
            target = target[selected_idxes]

            # filter according to max num bboxes
            _, idx = torch.sort(target.get_field('scores'), dim=0, descending=True)
            keep_boxes = idx[:min(detections_per_img, len(target))]
            target = target[keep_boxes]

            return target
        else:
            return []
    
    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_class[class_id]

    def map_attribute_id_to_attribute_name(self, attribute_id):
        return self.ind_to_attribute[attribute_id]

    def map_relation_id_to_relation_name(self, relation_id):
        return self.ind_to_relation[relation_id]
    
    def __getitem__(self, idx):
        img = self.get_image(idx)
        img_size = img.size # w, h
        annotations = self.get_annotations(idx)
        target = self.get_target_from_annotations(annotations, img_size)
        detection_boxlist = self.load_detection_result(idx, img_size, conf_threshold=self.predictedbbox_conf_threshold, detections_per_img=self.detections_per_img) if self.detector_pre_calculated else None
        img, target, detection_boxlist = self.apply_transforms(img, target, detection_boxlist)
        new_img_size = img.shape[1:]
        scale = math.sqrt(float(new_img_size[0]*new_img_size[1])/float(img_size[0]*img_size[1]))

        if self.relation_on and self.contrastive_loss_on and target is not None:
            self.contrastive_loss_target_transform(target)
        
        if self.detector_pre_calculated:
            return img, (target, detection_boxlist), idx, scale

        return img, target, idx, scale

    def contrastive_loss_target_transform(self, target):
        # add relationship annotations
        relation_triplets = target.get_field("relation_labels")
        sbj_gt_boxes = np.zeros((len(relation_triplets), 4), dtype=np.float32)
        obj_gt_boxes = np.zeros((len(relation_triplets), 4), dtype=np.float32)
        sbj_gt_classes_minus_1 = np.zeros(len(relation_triplets), dtype=np.int32)
        obj_gt_classes_minus_1 = np.zeros(len(relation_triplets), dtype=np.int32)
        prd_gt_classes_minus_1 = np.zeros(len(relation_triplets), dtype=np.int32)
        for ix, rel in enumerate(relation_triplets):
            # sbj
            sbj_gt_box = target.bbox[rel[0]]
            sbj_gt_boxes[ix] = sbj_gt_box
            # sbj_gt_classes_minus_1[ix] = rel['subject']['category']  # excludes background
            sbj_gt_classes_minus_1[ix] = target.get_field('labels')[rel[0]] - 1  # excludes background
            # obj
            obj_gt_box = target.bbox[rel[1]]
            obj_gt_boxes[ix] = obj_gt_box
            # obj_gt_classes_minus_1[ix] = rel['object']['category']  # excludes background
            obj_gt_classes_minus_1[ix] = target.get_field('labels')[rel[1]] - 1  # excludes background
            # prd
            # prd_gt_classes_minus_1[ix] = rel['predicate']  # excludes background
            prd_gt_classes_minus_1[ix] = rel[2] - 1  # excludes background

        target.add_field('sbj_gt_boxes', torch.from_numpy(sbj_gt_boxes))
        target.add_field('obj_gt_boxes', torch.from_numpy(obj_gt_boxes))
        target.add_field('sbj_gt_classes_minus_1', torch.from_numpy(sbj_gt_classes_minus_1))
        target.add_field('obj_gt_classes_minus_1', torch.from_numpy(obj_gt_classes_minus_1))
        target.add_field('prd_gt_classes_minus_1', torch.from_numpy(prd_gt_classes_minus_1))

        # misc
        num_obj_classes = len(self.class_to_ind) - 1  # excludes background
        num_prd_classes = len(self.relation_to_ind) - 1  # excludes background

        sbj_gt_overlaps = np.zeros(
            (len(relation_triplets), num_obj_classes), dtype=np.float32)
        for ix in range(len(relation_triplets)):
            sbj_cls = sbj_gt_classes_minus_1[ix]
            sbj_gt_overlaps[ix, sbj_cls] = 1.0
        # sbj_gt_overlaps = scipy.sparse.csr_matrix(sbj_gt_overlaps)
        target.add_field('sbj_gt_overlaps', torch.from_numpy(sbj_gt_overlaps))

        obj_gt_overlaps = np.zeros(
            (len(relation_triplets), num_obj_classes), dtype=np.float32)
        for ix in range(len(relation_triplets)):
            obj_cls = obj_gt_classes_minus_1[ix]
            obj_gt_overlaps[ix, obj_cls] = 1.0
        # obj_gt_overlaps = scipy.sparse.csr_matrix(obj_gt_overlaps)
        target.add_field('obj_gt_overlaps', torch.from_numpy(obj_gt_overlaps))

        prd_gt_overlaps = np.zeros(
            (len(relation_triplets), num_prd_classes), dtype=np.float32)
        pair_to_gt_ind_map = np.zeros(
            (len(relation_triplets)), dtype=np.int32)
        for ix in range(len(relation_triplets)):
            prd_cls = prd_gt_classes_minus_1[ix]
            prd_gt_overlaps[ix, prd_cls] = 1.0
            pair_to_gt_ind_map[ix] = ix
        # prd_gt_overlaps = scipy.sparse.csr_matrix(prd_gt_overlaps)
        target.add_field('prd_gt_overlaps', torch.from_numpy(prd_gt_overlaps))
        target.add_field('pair_to_gt_ind_map', torch.from_numpy(pair_to_gt_ind_map))
    