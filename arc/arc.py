import numpy as np
import time
import sys

from tools import findCandClips, calculate_reliability, calculate_start, calculate_end
from score_tools import entropy
from pruning_phase import init_probabilities, init_cluster_uncertainties
from refinement_phase import progressive_sampling, label_propagation, update_uncertainties, calculate_confidence


def progress_bar(current, total, tau_rel_confidence, tau_confidence, interval_o, bar_length=20):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    sys.stdout.write(
        f"\r sample/budget: [{''.join([arrow, padding])}] {current}/{total} ({fraction:.2%}) tau_rel_confidence: {tau_rel_confidence:.2f}, tau_confidence: {tau_confidence:.2f}, skip_round: {interval_o}")
    sys.stdout.flush()


def arc(proxy, oracle, proxy_score, oracle_score, B, op, constant, tau, confidence, IOUThreshold, clusters, startup_sampling_rate=0.002, tc_enabled=True, ps_enabled=True, lp_enabled=True):

    start_t = time.time()

    clusters = np.arange(0, len(proxy)) if not tc_enabled else clusters
    PD = proxy.copy()
    score = proxy_score.copy()
    n = len(proxy)
    init_probabilities(PD, 0, [0.9999, 0.0001])
    init_probabilities(PD, 1, [0.0001, 0.9999])

    entropy_array = np.apply_along_axis(entropy, 1, proxy)
    cluster_boundaries, uncertainties, probabilities = init_cluster_uncertainties(
        clusters, entropy_array, proxy)
    reliability = 0.5
    lb_t = 0
    o_b_true = 0
    frame_reward = np.zeros(n, dtype=int)
    sampled = np.zeros(n, dtype=bool)
    sampled_boundaries = np.full((n, 2), [-n, 2 * n], dtype=int)

    ce_start_t = time.time()
    tau_rel_cand_clips = findCandClips(score, '>', 0, tau*(1-reliability))
    _, tau_rel_confidence = calculate_confidence(
        tau_rel_cand_clips, IOUThreshold, probabilities, clusters, cluster_boundaries)
    tau_cand_clips = findCandClips(score, '>', 0, tau)
    _, tau_confidence = calculate_confidence(
        tau_cand_clips, IOUThreshold, probabilities, clusters, cluster_boundaries)
    interval_o = max(int((confidence-tau_rel_confidence)*len(tau_rel_cand_clips)),
                     int((confidence-tau_confidence)*len(tau_cand_clips)))
    interval_o = max(interval_o, 1)
    ce_end_t = time.time()
    ce_t = ce_end_t-ce_start_t

    progress_bar(0, B, tau_rel_confidence, tau_confidence, interval_o)

    for o_b in range(1, B):

        tau_rel_cand_clips = findCandClips(score, '>', 0, tau*(1-reliability))
        ce_start_t = time.time()
        if o_b % interval_o == 0:

            tau_cand_clips = findCandClips(score, op, constant, tau)
            _, tau_confidence = calculate_confidence(
                tau_cand_clips, IOUThreshold, probabilities, clusters, cluster_boundaries)

            if tau_confidence >= confidence:
                _, tau_rel_confidence = calculate_confidence(
                    tau_rel_cand_clips, IOUThreshold, probabilities, clusters, cluster_boundaries)
                if tau_rel_confidence >= confidence:
                    break

            interval_o = max(int((confidence-tau_rel_confidence)*len(tau_rel_cand_clips)),
                             int((confidence-tau_confidence)*len(tau_cand_clips)))
            interval_o = max(interval_o, 1)
            progress_bar(o_b + 1, B, tau_rel_confidence,
                         tau_confidence, interval_o)

        ce_end_t = time.time()
        ce_t = ce_t + ce_end_t-ce_start_t

        if ps_enabled == True:
            starts = np.array([calculate_start(a, b, IOUThreshold, tau)
                              for a, b in tau_rel_cand_clips])
            ends = np.array([calculate_end(a, b, IOUThreshold, tau, n)
                            for a, b in tau_rel_cand_clips])

            bool_cand = np.zeros(n, dtype=bool)
            for start, end in zip(starts, ends):
                bool_cand[start:end + 1] = True
            not_sampled = ~sampled
            bool_cand &= not_sampled
            bool_non_cand = not_sampled & ~bool_cand

            max_i = progressive_sampling(reliability, bool_non_cand, not_sampled, entropy_array,
                                         starts, ends, sampled, frame_reward, uncertainties, o_b, startup_sampling_rate*B)

        else:
            not_sampled = ~sampled
            valid_indices = np.where(not_sampled)[0]
            if len(valid_indices) != 0:
                max_i = np.random.choice(valid_indices)
            else:
                break

        if max_i is None:
            break

        if o_b < startup_sampling_rate * B:
            if score[max_i] == oracle_score[max_i]:
                o_b_true += 1
            reliability = calculate_reliability(o_b_true, o_b + 1)

        lb_t_update, left, right, leftT, rightT = label_propagation(
            PD, score, sampled, cluster_boundaries, sampled_boundaries, max_i, tau, oracle, probabilities, clusters, frame_reward, uncertainties, lp_enabled=lp_enabled)
        lb_t += lb_t_update

        if ps_enabled == True and lp_enabled == True and score[max_i] == 0:
            leftT = max(0, leftT, left - tau)
            rightT = min(rightT, len(proxy) - 1, right + tau)
            update_uncertainties(leftT + 1, left, uncertainties, clusters, probabilities,
                                 tau, score, cluster_boundaries, leftT, left, entropy_array)
            update_uncertainties(right + 1, rightT - 1, uncertainties, clusters,
                                 probabilities, tau, score, cluster_boundaries, right, rightT, entropy_array)

    tau_cand_clips = findCandClips(score, '>', 0, tau)
    _, tau_confidence = calculate_confidence(
        tau_cand_clips, IOUThreshold, probabilities, clusters, cluster_boundaries)
    progress_bar(o_b + 1, B, tau_rel_confidence, tau_confidence, interval_o)
    end_t = time.time()

    return {
        'cand_clips': tau_cand_clips, 'B': o_b + 1, 'ps_t': end_t - start_t - lb_t, 'lb_t': lb_t, 'ce_t': ce_t, 'tc_enabled': tc_enabled
    }
