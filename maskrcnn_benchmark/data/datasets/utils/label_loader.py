# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import torch
import numpy as np
import base64
import collections

from maskrcnn_benchmark.structures.bounding_box import BoxList


class LabelLoader(object):
    def __init__(self, labelmap, extra_fields=(), filter_duplicate_relations=False, ignore_attr=None, ignore_rel=None):
        self.labelmap = labelmap
        self.extra_fields = extra_fields
        self.supported_fields = ["class", "conf", "attributes", 'scores_all', 'boxes_all', 'feature']
        self.filter_duplicate_relations = filter_duplicate_relations
        self.ignore_attr = set(ignore_attr) if ignore_attr != None else set()
        self.ignore_rel = set(ignore_rel) if ignore_rel != None else set()

    def __call__(self, annotations, img_size, remove_empty=False, load_fields=None):
        boxes = [obj["rect"] for obj in annotations]
        boxes = torch.as_tensor(boxes).reshape(-1, 4) 
        target = BoxList(boxes, img_size, mode="xyxy")

        if load_fields is None:
            load_fields = self.extra_fields

        for field in load_fields:
            assert field in self.supported_fields, "Unsupported field {}".format(field)
            if field == "class":
                classes = self.add_classes(annotations)
                target.add_field("labels", classes)
            elif field == "conf":
                confidences = self.add_confidences(annotations)
                target.add_field("scores", confidences)
            elif field == "attributes":
                attributes = self.add_attributes(annotations)
                target.add_field("attributes", attributes)
            elif field == "scores_all":
                scores_all = self.add_scores_all(annotations)
                target.add_field("scores_all", scores_all)
            elif field == "boxes_all":
                boxes_all = self.add_boxes_all(annotations)
                target.add_field("boxes_all", boxes_all)
            elif field == "feature":
                features = self.add_features(annotations)
                target.add_field("box_features", features)
                
        target = target.clip_to_image(remove_empty=remove_empty)
        return target

    def add_classes(self, annotations):
        class_names = [obj["class"] for obj in annotations]
        classes = [None] * len(class_names)       
        for i in range(len(class_names)):
            classes[i] = self.labelmap['class_to_ind'][class_names[i]]
        return torch.tensor(classes)

    def add_confidences(self, annotations):
        confidences = []
        for obj in annotations:
            if "conf" in obj:
                confidences.append(obj["conf"])
            else:
                confidences.append(1.0)
        return torch.tensor(confidences)

    def add_attributes(self, annotations):
        # the maximal number of attributes per object is 16
        attributes = [ [0] * 16 for _ in range(len(annotations))]
        for i, obj in enumerate(annotations):
            for j, attr in enumerate(obj["attributes"]):
                attributes[i][j] = self.labelmap['attribute_to_ind'][attr]
        return torch.tensor(attributes)

    def add_features(self, annotations):
        features = []
        for obj in annotations:
            features.append(np.frombuffer(base64.b64decode(obj['feature']), np.float32))
        return torch.tensor(features)
    
    def add_scores_all(self, annotations):
        scores_all = []
        for obj in annotations:
            scores_all.append(np.frombuffer(base64.b64decode(obj['scores_all']), np.float32))
        return torch.tensor(scores_all)
    
    def add_boxes_all(self, annotations):
        boxes_all = []
        for obj in annotations:
            boxes_all.append(np.frombuffer(base64.b64decode(obj['boxes_all']), np.float32).reshape(-1, 4))
        return torch.tensor(boxes_all)
    
    def relation_loader(self, relation_annos, target):
        if self.filter_duplicate_relations:
            # Filter out dupes!
            all_rel_sets = collections.defaultdict(list)
            for triplet in relation_annos:
                all_rel_sets[(triplet['subj_id'], triplet['obj_id'])].append(triplet)
            relation_annos = [np.random.choice(v) for v in all_rel_sets.values()]

        # get M*M pred_labels
        relation_triplets = []
        relations = torch.zeros([len(target), len(target)], dtype=torch.int64)
        for i in range(len(relation_annos)):
            if len(self.ignore_rel)!=0 and relation_annos[i]['class'] in self.ignore_rel:
                continue
            subj_id = relation_annos[i]['subj_id']
            obj_id = relation_annos[i]['obj_id']
            predicate = self.labelmap['relation_to_ind'][relation_annos[i]['class']]
            relations[subj_id, obj_id] = predicate
            relation_triplets.append([subj_id, obj_id, predicate])

        relation_triplets = torch.tensor(relation_triplets)
        target.add_field("relation_labels", relation_triplets)
        target.add_field("pred_labels", relations)
        return target