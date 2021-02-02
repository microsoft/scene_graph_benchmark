# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import numpy as np


def is_valid_rect(rc):
    # a rect is valid if x2 > x1 and y2 > y1
    return rc[2] > rc[0] and rc[3] > rc[1]


def rect_area(rc):
    return (float(rc[2]) - rc[0]) * (rc[3] - rc[1])


def IoU(rc1, rc2):
    rc_inter = [max(rc1[0], rc2[0]), max(rc1[1], rc2[1]), 
            min(rc1[2], rc2[2]), min(rc1[3], rc2[3])]
    if is_valid_rect(rc_inter):
        return rect_area(rc_inter) / (rect_area(rc1) + rect_area(rc2) 
                                      - rect_area(rc_inter))
    return 0


def IoA(rc1, rc2):
    """ Intersection over smaller box area, used in group-of box evaluation.
    Args: 
        rc1: A list of the smaller box coordinates in xyxy mode
        rc2: A list of the group box coordinates in xyxy mode
    Returns:
        ioa: A float number of ioa score = intersection(rc1, rc2) / area(rc1)
    """
    rc_inter = [max(rc1[0], rc2[0]), max(rc1[1], rc2[1]), 
                min(rc1[2], rc2[2]), min(rc1[3], rc2[3])]
    if is_valid_rect(rc_inter):
        return rect_area(rc_inter) / rect_area(rc1)
    else:
        return 0


def get_overlaps(det, gt):
    """ Calculate IoU and IoA for a list of detected and ground-truth boxes. 
    Args: 
        det: A list of D detection results (from det_dict[label][key])
        gt: A list of G ground-truth results (from gt_dict[label][key]),
            and say there are G1 group-of box and G2 non group-of box
        
    Returns:
        ious: A float numpy array (D*G1) of IoU scores between detection 
              and non group-of ground-truth boxes
        ioas: A float numpy array (D*G2) of IoA scores between detection 
              and group-of ground-truth boxes
    """
    gt_is_group = [g for g in gt if g[0]!=0]
    gt_is_non_group = [g for g in gt if g[0]==0] 
    ious = [[IoU(d[1], g[1]) for g in gt_is_non_group] for d in det]
    ioas = [[IoA(d[1], g[1]) for g in gt_is_group] for d in det]
    return np.array(ious), np.array(ioas)


def compute_precision_recall(scores, labels, num_gt):
    assert np.sum(labels) <= num_gt, \
            "number of true positives must be no larger than num_gt."
    assert len(scores) == len(labels), \
            "scores and labels must be the same size."
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    tp_labels = labels[sorted_indices]
    fp_labels = (tp_labels <= 0).astype(float)
    cum_tp = np.cumsum(tp_labels)
    cum_fp = np.cumsum(fp_labels)
    precision = cum_tp.astype(float) / (cum_tp + cum_fp)
    recall = cum_tp.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    if not precision.size:
        return 0.0
    assert len(precision) == len(recall), \
           "precision and recall must be of the same size."
    assert np.amin(precision) >= 0 and np.amax(precision) <= 1, \
           "precision must be in the range of [0, 1]."
    assert np.amin(recall) >= 0 and np.amax(recall) <= 1, \
           "recall must be in the range of [0, 1]."
    assert all(recall[i] <= recall[i+1] for i in range(len(recall)-1)), \
           "recall must be a non-decreasing array"

    rec = np.concatenate([[0], recall, [1]])
    prec = np.concatenate([[0], precision, [0]])
    # pre-process precision to be a non-decreasing array
    for i in range(len(prec) - 2, -1, -1):
      prec[i] = np.maximum(prec[i], prec[i + 1])
    indices = np.where(rec[1:] != rec[:-1])[0] + 1
    ap = np.sum((rec[indices] - rec[indices - 1]) * prec[indices])
    return ap


def compute_recall_at_k(tp_fp_list, num_gt, k):
    """Computes Recall@k, MedianRank@k, where k is the top-scoring labels.
    Args:
        tp_fp_list: a list of numpy arrays; each numpy array corresponds to the all
            detection on a single image, where the detections are sorted by score in
            descending order. Further, each numpy array element can have boolean or
            float values. True positive elements have either value >0.0 or True;
            any other value is considered false positive.
        num_gt: number of groundtruth anotations.
        k: number of top-scoring proposals to take.
    Returns:
        recall: recall evaluated on the top k by score detections.
    """

    tp_fp_eval = []
    for i in range(len(tp_fp_list)):
        tp_fp_eval.append(tp_fp_list[i][:min(k, len(tp_fp_list[i]))])

    tp_fp_eval = np.concatenate(tp_fp_eval)

    return np.sum(tp_fp_eval) / num_gt