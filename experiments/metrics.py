import numpy as np
from experiment_config import TIME_CONFIG
from results_io import save_results


def calculate_iou(segment1, segment2):

    intersection_start = np.maximum(segment1[0], segment2[0])
    intersection_end = np.minimum(segment1[1], segment2[1])
    intersection_length = np.maximum(
        0, intersection_end - intersection_start + 1)

    union_start = np.minimum(segment1[0], segment2[0])
    union_end = np.maximum(segment1[1], segment2[1])
    union_length = union_end - union_start + 1

    return intersection_length / union_length


def calculate_precision_recall_iou(gt_segments, approx_segments, threshold):

    hits_precision = 0
    hits_recall = 0
    total_iou = 0.0

    for approx_segment in approx_segments:
        ious = np.array([calculate_iou(approx_segment, gt_segment)
                        for gt_segment in gt_segments])
        if np.any(ious >= threshold):
            hits_precision += 1

    for gt_segment in gt_segments:
        ious = np.array([calculate_iou(gt_segment, approx_segment)
                        for approx_segment in approx_segments])
        max_iou = np.max(ious, initial=0)
        if max_iou >= threshold:
            hits_recall += 1
        total_iou += max_iou
    precision = hits_precision / \
        len(approx_segments) if len(approx_segments) > 0 else 0
    recall = hits_recall / len(gt_segments) if len(gt_segments) > 0 else 0
    average_iou = total_iou / len(gt_segments) if len(gt_segments) > 0 else 0

    return precision, recall, average_iou


def calculate_time(oracle_calls, proxy_calls, YOLOv5s_calls, V, tag=0, ps_t=0, lb_t=0, ce_t=0, tc_enabled=True):
    tc_count = 1 if tc_enabled else 0
    time_ee = (TIME_CONFIG["oracle_time"][V] * oracle_calls +
               TIME_CONFIG["proxy_time"][V] * proxy_calls +
               TIME_CONFIG["YOLOv5s_time"][V] * YOLOv5s_calls +
               tag * TIME_CONFIG["cluster_time"][V] * tc_count +
               ps_t + lb_t)
    '''
    if tag == 1:
        time_details = {
            "V": V,
            "Initial processing": TIME_CONFIG["proxy_time"][V] * proxy_calls / time_ee,
            "Time-domain clustering": TIME_CONFIG["cluster_time"][V]*tc_count / time_ee,
            "Progressive sampling": (ps_t + TIME_CONFIG["oracle_time"][V] * oracle_calls) / time_ee,
            "Label propagation": lb_t / time_ee,
            "Confidence Estimation": ce_t / time_ee,
            "Sampling": (ps_t + TIME_CONFIG["oracle_time"][V] * oracle_calls-ce_t) / time_ee,
        }
        save_results(time_details, f"{V}_time_breakdown.pkl")
    '''
    return time_ee
