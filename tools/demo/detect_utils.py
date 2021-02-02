# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import cv2
import torch
from PIL import Image

from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]

    if isinstance(model, SceneParser):
        prediction_pred = prediction.prediction_pairs
        relations = prediction_pred.get_field("idx_pairs").tolist()
        relation_scores = prediction_pred.get_field("scores").tolist()
        predicates = prediction_pred.get_field("labels").tolist()
        prediction = prediction.predictions

    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()

    if isinstance(model, SceneParser):
        rt_box_list = []
        if 'attr_scores' in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist()}
                for box, cls, score, attr, attr_conf in
                zip(boxes, classes, scores, attr_labels, attr_scores)
            ]
        else:
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score}
                for box, cls, score in
                zip(boxes, classes, scores)
            ]
        rt_relation_list = [{"subj_id": relation[0], "obj_id":relation[1], "class": predicate+1, "conf": score}
                for relation, predicate, score in
                zip(relations, predicates, relation_scores)]
        return {'objects': rt_box_list, 'relations':rt_relation_list}
    else:
        if 'attr_scores' in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            return [
                {"rect": box, "class": cls, "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist()}
                for box, cls, score, attr, attr_conf in
                zip(boxes, classes, scores, attr_labels, attr_scores)
            ]

        return [
            {"rect": box, "class": cls, "conf": score}
            for box, cls, score in
            zip(boxes, classes, scores)
        ]

