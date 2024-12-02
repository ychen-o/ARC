from cmdn_aqp import uniform, importance
from supg import run_rt, run_pt
from tools import findCandClips
from arc import arc
import numpy as np
from metrics import calculate_time, calculate_precision_recall_iou
import sys
sys.path.append('../arc')


def oracle_only(oracle_score, op, tau):
    oracle_score_2 = oracle_score.copy()
    cand_clips = findCandClips(oracle_score_2, op, 0, tau)
    oracle_calls = len(oracle_score_2)
    return cand_clips, oracle_calls


def yolo_only(yolo_oracle_score, op, tau):
    yolo_oracle_score_2 = yolo_oracle_score.copy()
    cand_clips = findCandClips(yolo_oracle_score_2, op, 0, tau)
    YOLOv5s_calls = len(yolo_oracle_score_2)
    return cand_clips, YOLOv5s_calls


def cmdn(name, function, proxy, oracle, proxy_score, oracle_score, B, op, tau):
    Algres = function(proxy, oracle, proxy_score, oracle_score, B, op, 0, tau)
    cand_clips = Algres['cand_clips']
    oracle_calls = Algres['B']
    proxy_calls = len(proxy)
    return cand_clips, oracle_calls, proxy_calls


def supg(name, supg_path, oracle, oracle_score, B, op, tau, target):
    sample4supg = int(B / 2)
    if name == 'SUPGrt+':
        _, inds = run_rt(supg_path, sample4supg, target)
    else:
        _, inds = run_pt(supg_path, sample4supg, target)
    supg_score = np.zeros(len(oracle))
    supg_score[inds] = 1
    sampled4remain = min(len(list(inds)), B - sample4supg)
    sample_remain_indices = np.random.choice(
        list(inds), sampled4remain, replace=False)
    supg_score[sample_remain_indices] = oracle_score[sample_remain_indices]
    cand_clips = findCandClips(supg_score, op, 0, tau)
    oracle_calls = sample4supg + sampled4remain
    proxy_calls = len(oracle)
    return cand_clips, oracle_calls, proxy_calls


def _arc(function, proxy, oracle, proxy_score, oracle_score, B, op, tau, confidence, IOUThreshold, clusters):
    Algres = function(proxy, oracle, proxy_score, oracle_score,
                      B, '>', 0, tau, confidence, IOUThreshold, clusters)
    cand_clips = Algres['cand_clips']
    oracle_calls = Algres['B']
    proxy_calls = len(proxy)
    return cand_clips, oracle_calls, proxy_calls, Algres


def run_algorithm(name, function, V, ground_truth, proxy, oracle, proxy_score, oracle_score, B, op, tau, IOUThreshold, confidence, clusters, yolo_oracle_score=None, supg_path=None):
    oracle_calls, proxy_calls, YOLOv5s_calls = 0, 0, 0

    if name == 'Oracle-Only':
        cand_clips, oracle_calls = oracle_only(oracle_score, op, tau)
    elif name == 'YOLOv5s-Only':
        cand_clips, YOLOv5s_calls = yolo_only(yolo_oracle_score, op, tau)
    elif name in ['CMDN-Uniform', 'CMDN-Importance']:
        cand_clips, oracle_calls, proxy_calls = cmdn(
            name, function, proxy, oracle, proxy_score, oracle_score, B, op, tau)
    elif name in ['SUPGrt+', 'SUPGpt+']:
        cand_clips, oracle_calls, proxy_calls = supg(
            name, supg_path, oracle, oracle_score, B, op, tau, confidence)
    elif name == 'ARC':
        cand_clips, oracle_calls, proxy_calls, Algres = _arc(
            function, proxy, oracle, proxy_score, oracle_score, B, op, tau, confidence, IOUThreshold, clusters)
        time_1 = calculate_time(oracle_calls, proxy_calls, YOLOv5s_calls, V, tag=1,
                                ps_t=Algres['ps_t'], lb_t=Algres['lb_t'], ce_t=Algres['ce_t'], tc_enabled=Algres['tc_enabled'])
        precision, recall, averageIOU = calculate_precision_recall_iou(
            ground_truth, cand_clips, IOUThreshold)

    if name not in ['ARC']:
        time_1 = calculate_time(oracle_calls, proxy_calls, YOLOv5s_calls, V)
        precision, recall, averageIOU = calculate_precision_recall_iou(
            ground_truth, cand_clips, IOUThreshold)

    return time_1, averageIOU, recall, precision
