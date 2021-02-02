# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import json
import os
import os.path as op
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from .eval_utils import IoU, compute_precision_recall, \
        compute_average_precision, is_valid_rect, compute_recall_at_k


def do_openimages_vrd_evaluation(dataset, output_folder, iou_thresh=0.5, force_relation=False):
    # output_folder could be a folder or the prediction tsv file
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Perform openimages visual relation evaluation.")
    # note the ground truth labels should be expanded beforehand
    logger.info("Prepare ground truth annotations.")
    img_gt_dict, triplet_gt_dict, phrase_gt_dict = prepare_vrd_groundtruths(dataset)

    prediction_file = op.join(output_folder, 'predictions.tsv')

    # # perform bounding box all proposal coverage evaluation, only need once per detector
    # evaluate_box_proposals_for_relation(prediction_file, dataset)
 
    logger.info("Prepare detection results.")
    pred_triplet_dict, pred_phrase_dict = prepare_vrd_predictions(prediction_file, img_gt_dict)

    logger.info("Start triplet and recall evaluation.")
    relation_eval(triplet_gt_dict, pred_triplet_dict, output_folder, eval_phrase=False, force_relation=force_relation)

    logger.info("Start phrase evaluation.")
    relation_eval(phrase_gt_dict, pred_phrase_dict, output_folder, eval_phrase=True, force_relation=force_relation)


def relation_eval(gt_dict, pred_dict, output_folder, eval_phrase=False, force_relation=False):
    scores_per_class = defaultdict(list)
    tp_fp_labels_per_class = defaultdict(list)
    num_gt_per_class = defaultdict(int)
    scores_per_img = defaultdict(list)
    tp_fp_labels_per_img = defaultdict(list)

    for cls in gt_dict.keys() | pred_dict.keys():
        c_truths = gt_dict[cls]
        c_dets = pred_dict[cls]
        scores, tp_fp_labels, num_gt, img_keys = eval_per_class(c_dets, c_truths, eval_phrase=eval_phrase)
        scores_per_class[cls[2]] += scores
        tp_fp_labels_per_class[cls[2]] += tp_fp_labels
        num_gt_per_class[cls[2]] += num_gt

        if not eval_phrase:
            for key, score, tp_fp in zip(img_keys, scores, tp_fp_labels):
                scores_per_img[key] += score.tolist()
                tp_fp_labels_per_img[key] += tp_fp.tolist()

    class_ap = {}
    for cls in scores_per_class.keys():
        if num_gt_per_class[cls]==0 or len(scores_per_class[cls])==0:
            continue
        scores = np.concatenate(scores_per_class[cls])
        tp_fp_labels = np.concatenate(tp_fp_labels_per_class[cls])
        precision, recall = compute_precision_recall(
                scores, tp_fp_labels, num_gt_per_class[cls])
        ap = compute_average_precision(precision, recall)
        class_ap[cls] = ap

    mean_ap = sum([class_ap[cls] for cls in class_ap]) / len(class_ap)
    total_gt = sum(num_gt_per_class.values())

    result_table = {"obj": {"map": mean_ap, "weighted map": sum([class_ap[cls]*num_gt_per_class[cls] for cls in class_ap])/total_gt, 'categories': class_ap}}
    if not eval_phrase:
        for key in scores_per_img.keys():
            tp_fp_labels_per_img[key] = [x for _, x in sorted(zip(scores_per_img[key], tp_fp_labels_per_img[key]), key=lambda x:x[0], reverse=True)]
        recall = compute_recall_at_k(list(tp_fp_labels_per_img.values()), total_gt, k=50)
        print('Recall at 50: ', recall)
        result_table['obj']['recall@50'] = recall

        if force_relation:
            tp_fps = np.concatenate(list(tp_fp_labels_per_img.values()))
            accuracy = np.sum(tp_fps) / len(tp_fps)
            print('Predicate classification Accuracy: ', accuracy)
            result_table['obj']['Predicate_classification_arruracy'] = accuracy
    
    rel_proposal_recall, rel_proposal_ap = relation_proposal_recall(gt_dict, pred_dict, eval_phrase=eval_phrase)
    result_table['obj']['relation_proposal_recall'] = rel_proposal_recall
    result_table['obj']['relation_proposal_ap'] = rel_proposal_ap

    output_file = op.join(output_folder, 'phrase_evaluation.json') if eval_phrase else op.join(output_folder, 'triplet_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(result_table, f)


def eval_per_class(c_dets, c_truths, overlap_thresh=0.5, eval_phrase=False):
    """ Evaluation for each class. 
    Args:
        c_dets: A dictionary of all detection results.
        c_truths: A dictionary of all ground-truth annotations.
        overlap_thresh: A float of the threshold used in IoU matching.

    Returns:
        scores_all: A list of numpy float array collecting the confidence scores 
                    of both truth positives and false positives in each image.
        tp_fp_labels_all: A list of numpy float array collecting the true 
                positives (=1) and false positives (=0) labels in each image.
        num_gt_all: An integer of the total number of valid ground-truth boxes.
    """
    num_gt_all = sum([len(c_truths[l]) for l in c_truths])

    scores_all = []
    tp_fp_labels_all = []
    img_keys = []
    for key in c_dets:
        img_keys.append(key)
        img_det = c_dets[key]
        num_det = len(img_det)
        scores = np.array([det['score'] for det in img_det])
        tp_fp_labels = np.zeros(num_det, dtype=bool)

        if key not in c_truths or all(scores<0):
            # detections not in ground truth or detections have negative image level label, classified as false positives
            scores_all.append(scores)
            tp_fp_labels_all.append(tp_fp_labels)
            continue

        img_gt = c_truths[key]

        if eval_phrase:
            ious = np.array([[IoU(d['rect'], g['rect']) for g in img_gt] for d in img_det])
        else:
            ious = np.array([[min(IoU(d['subject_rect'], g['subject_rect']), IoU(d['object_rect'], g['object_rect'])) for g in img_gt] for d in img_det])
        if ious.shape[1] > 0:
            max_overlap_gt_ids = np.argmax(ious, axis=1)
            is_gt_box_detected = np.zeros(ious.shape[1], dtype=bool)
            for i in range(num_det):
                gt_id = max_overlap_gt_ids[i]
                if ious[i, gt_id] >= overlap_thresh:
                    if not is_gt_box_detected[gt_id]:
                        tp_fp_labels[i] = True
                        is_gt_box_detected[gt_id] = True
        
        # if ious.shape[1] > 0:
        #     max_overlap_gt_ids = np.argsort(-1*ious, axis=1)
        #     is_gt_box_detected = np.zeros(ious.shape[1], dtype=bool)
        #     for i in range(num_det):
        #         for gt_id in max_overlap_gt_ids[i, :]:
        #             if ious[i, gt_id] >= overlap_thresh:
        #                 if not is_gt_box_detected[gt_id]:
        #                     tp_fp_labels[i] = True
        #                     is_gt_box_detected[gt_id] = True
        #                     break
        #             else:
        #                 break
        
        # num_gt = len(img_gt)
        # if ious.shape[1] > 0:
        #     max_overlap_det_ids = np.argsort(-1*ious, axis=0)
        #     is_det_box_used = np.zeros(ious.shape[0], dtype=bool)
        #     for i in range(num_gt):
        #         for det_id in max_overlap_det_ids[:, i]:
        #             if ious[det_id, i] >= overlap_thresh:
        #                 if not is_det_box_used[det_id]:
        #                     tp_fp_labels[det_id] = True
        #                     is_det_box_used[det_id] = True
        #                     break
        #             else:
        #                 break

        scores_all.append(scores)
        tp_fp_labels_all.append(tp_fp_labels)
 
    return scores_all, tp_fp_labels_all, num_gt_all, img_keys


def prepare_vrd_groundtruths(dataset):
    img_gt_dict = defaultdict(dict)
    triplet_gt_dict = defaultdict(lambda:defaultdict(list))
    phrase_gt_dict = defaultdict(lambda:defaultdict(list))
    for idx in range(len(dataset)):
        img_key = dataset.get_img_key(idx)
        img_label = dataset.get_imagelabel(idx)
        label = dataset.get_annotations(idx)

        for obj in label['objects']:
            cls = obj['class']
            img_gt_dict[cls][img_key] = 1
        for obj in img_label:
            cls = obj['class']
            if obj['conf'] == 0:
            # if img_key not in img_gt_dict[cls]: # all verified image level labels are negative except those in bbox
                img_gt_dict[cls][img_key] = -1
            else:
                img_gt_dict[cls][img_key] = 1

        bboxes = label['objects']
        
        for triplet in label['relations']:
            subj_id = triplet['subj_id']
            obj_id = triplet['obj_id']
            xmin = min(bboxes[subj_id]['rect'][0], bboxes[obj_id]['rect'][0])
            ymin = min(bboxes[subj_id]['rect'][1], bboxes[obj_id]['rect'][1])
            xmax = max(bboxes[subj_id]['rect'][2], bboxes[obj_id]['rect'][2])
            ymax = max(bboxes[subj_id]['rect'][3], bboxes[obj_id]['rect'][3])
            class_name = (bboxes[subj_id]['class'], bboxes[obj_id]['class'], triplet['class'])
            phrase = {'rect':[xmin, ymin, xmax, ymax]}
            phrase_gt_dict[class_name][img_key].append(phrase)
            triplet_gt_dict[class_name][img_key].append({'subject_rect':bboxes[subj_id]['rect'], 'object_rect':bboxes[obj_id]['rect']})
        
    return img_gt_dict, triplet_gt_dict, phrase_gt_dict
    

def prepare_vrd_predictions(pred_tsv_file, img_gt_dict):
    filtered_triplet_dict = defaultdict(lambda:defaultdict(list))
    filtered_phrase_dict = defaultdict(lambda:defaultdict(list))
    for row in tqdm(tsv_reader(pred_tsv_file)):
        img_key = row[0]
        predictions = json.loads(row[1])
        for triplet in predictions['relations']:
            subj = predictions['objects'][triplet['subj_id']]
            obj = predictions['objects'][triplet['obj_id']]
            if True or ((subj['class'] in img_gt_dict and img_key in img_gt_dict[subj['class']]) and \
                (obj['class'] in img_gt_dict and img_key in img_gt_dict[obj['class']])) or \
                (subj['class'] in img_gt_dict and img_key in img_gt_dict[subj['class']] and img_gt_dict[subj['class']][img_key]==-1) or \
                (obj['class'] in img_gt_dict and img_key in img_gt_dict[obj['class']] and img_gt_dict[obj['class']][img_key] == -1):
                class_name = (subj['class'], obj['class'], triplet['class'])
                filtered_triplet_dict[class_name][img_key].append({'subject_rect':subj['rect'], 'object_rect':obj['rect'], 'score':triplet['conf']})
                xmin = min(subj['rect'][0], obj['rect'][0])
                ymin = min(subj['rect'][1], obj['rect'][1])
                xmax = max(subj['rect'][2], obj['rect'][2])
                ymax = max(subj['rect'][3], obj['rect'][3])
                phrase = {'rect':[xmin, ymin, xmax, ymax], 'score':triplet['conf']}
                filtered_phrase_dict[class_name][img_key].append(phrase)

    # sort the results based on confidence score (high to low)
    for cls in filtered_triplet_dict:
        for img_key in filtered_triplet_dict[cls]:
            filtered_triplet_dict[cls][img_key] = sorted(filtered_triplet_dict[cls][img_key], key=lambda x:x['score'], reverse=True)
    for cls in filtered_phrase_dict:
        for img_key in filtered_phrase_dict[cls]:
            filtered_phrase_dict[cls][img_key] = sorted(filtered_phrase_dict[cls][img_key], key=lambda x:x['score'], reverse=True)
    return filtered_triplet_dict, filtered_phrase_dict


def relation_proposal_recall(gt_dict, pred_dict, eval_phrase=False):
    gt_pairs = defaultdict(lambda:defaultdict(list))
    for subj_obj_rel_class, c_gts in gt_dict.items():
        for img_key, trips in c_gts.items():
            subj_obj_class = (subj_obj_rel_class[0], subj_obj_rel_class[1])
            gt_pairs[subj_obj_class][img_key] += trips

    pred_pairs = defaultdict(lambda:defaultdict(list))
    for subj_obj_rel_class, c_preds in pred_dict.items():
        for img_key, trips in c_preds.items():
            subj_obj_class = (subj_obj_rel_class[0], subj_obj_rel_class[1])
            pred_pairs[subj_obj_class][img_key] += trips
    for cls in pred_pairs:
        for img_key in pred_pairs[cls]:
            pred_pairs[cls][img_key].sort(key=lambda x:x['score'], reverse=True)

    total_gt = 0
    scores_per_img = defaultdict(list)
    tp_fp_labels_per_img = defaultdict(list)
    for cls in gt_pairs.keys() | pred_pairs.keys():
        c_truths = gt_pairs[cls]
        c_dets = pred_pairs[cls]
        scores, tp_fp_labels, num_gt, img_keys = eval_per_class(c_dets, c_truths, eval_phrase=eval_phrase)
        total_gt += num_gt
        for key, score, tp_fp in zip(img_keys, scores, tp_fp_labels):
            scores_per_img[key] += score.tolist()
            tp_fp_labels_per_img[key] += tp_fp.tolist()
    
    for key in scores_per_img.keys():
        tp_fp_labels_per_img[key] = [x for _, x in sorted(zip(scores_per_img[key], tp_fp_labels_per_img[key]), key=lambda x:x[0], reverse=True)]
    # calculate recall using all pairs
    recall = compute_recall_at_k(list(tp_fp_labels_per_img.values()), total_gt, k=1e10)

    scores = np.concatenate(list(scores_per_img.values()))
    tp_fp_labels = np.concatenate(list(tp_fp_labels_per_img.values()))
    precision_list, recall_list = compute_precision_recall(
            scores, tp_fp_labels, total_gt)
    ap = compute_average_precision(precision_list, recall_list)
    
    if not eval_phrase:
        print("Triplet relation proposal recall: ", recall)
        print("Triplet relation proposal ap: ", ap)
    else:
        print("Phrase relation proposal recall: ", recall)
        print("Phrase relation proposal ap: ", ap)
    return recall, ap


def evaluate_box_proposals_for_relation(pred_tsv_file, dataset):
    from itertools import combinations 
    pred_triplet_dict = defaultdict(lambda:defaultdict(list))
    pred_phrase_dict = defaultdict(lambda:defaultdict(list))
    for row in tqdm(tsv_reader(pred_tsv_file)):
        img_key = row[0]
        predictions = json.loads(row[1])

        pairs = list(combinations(range(len(predictions['objects'])), 2))
        pairs += [(pair[1], pair[0]) for pair in pairs]
        for pair in pairs:
            subj = predictions['objects'][pair[0]]
            obj = predictions['objects'][pair[1]]

            class_name = (subj['class'], obj['class'])
            pred_triplet_dict[class_name][img_key].append({'subject_rect':subj['rect'], 'object_rect':obj['rect'], 'score':1.0})
            xmin = min(subj['rect'][0], obj['rect'][0])
            ymin = min(subj['rect'][1], obj['rect'][1])
            xmax = max(subj['rect'][2], obj['rect'][2])
            ymax = max(subj['rect'][3], obj['rect'][3])
            phrase = {'rect':[xmin, ymin, xmax, ymax], 'score':1.0}
            pred_phrase_dict[class_name][img_key].append(phrase)
    
    img_gt_dict, triplet_gt_dict, phrase_gt_dict = prepare_vrd_groundtruths(dataset)

    print('Evaluate box proposal for triplet relation: ')
    triplet_all_proposal_recall, triplet_all_proposal_ap = relation_proposal_recall(triplet_gt_dict, pred_triplet_dict, eval_phrase=False)
    print('Evaluate box proposal for phrase relation: ')
    phrase_all_proposal_recall, phrase_all_proposal_ap = relation_proposal_recall(phrase_gt_dict, pred_phrase_dict, eval_phrase=True)
