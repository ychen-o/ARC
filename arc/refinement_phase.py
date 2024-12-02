import numpy as np
import time
from sampling_tools import select_indices_uniform, select_indices_importance


def progressive_sampling(reliability, bool_non_cand, not_sampled, entropy_array, starts, ends, sampled, frame_reward, uncertainties, o_b, o_b_s):
    if o_b < o_b_s:
        true_indices = np.where(not_sampled)[0]
        return select_indices_uniform(true_indices, 1)[0]
    else:
        if np.random.rand() < reliability:
            return compute_ucb(starts, ends, sampled, frame_reward, uncertainties, o_b)
        else:
            true_indices = np.where(bool_non_cand)[0]
            selected_result = select_indices_importance(
                true_indices, entropy_array[true_indices], 1)
            if selected_result is None:
                return None
            return selected_result[0]


def compute_ucb(starts, ends, sampled, frame_reward, uncertainties, o_b):
    all_sampled_mask = np.array(
        [np.all(sampled[start:end + 1]) for start, end in zip(starts, ends)])
    if not any(~all_sampled_mask):
        return None
    active_starts = starts[~all_sampled_mask]
    active_ends = ends[~all_sampled_mask]
    ucb_values = []

    for start, end in zip(active_starts, active_ends):
        rewards = frame_reward[start:end + 1]
        non_zero_rewards = rewards[rewards != 0]
        ni = len(non_zero_rewards)
        reward = np.mean(non_zero_rewards) if ni > 0 else np.max(
            uncertainties[start:end + 1])
        ucb = reward + 2 * np.sqrt(2 * np.log(o_b + 1) / ni) if ni > 0 else 0
        ucb_values.append((ucb, start, end))
    max_ucb, best_start, best_end = max(ucb_values)
    return np.argmax(uncertainties[best_start:best_end + 1]) + best_start


def adjust_boundary(current, target, tau, score, decrease=False):
    if decrease:
        if (target - current) <= tau and score[target] == 0:
            return target - 1
    else:
        if (current - target) <= tau and score[target] == 0:
            return target + 1
    return current


def apply_updates(PD, score, sampled, left, right, max_i):
    PD[left:right+1] = PD[max_i]
    score[left:right+1] = score[max_i]
    sampled[left:right+1] = True


def update_sampled_boundaries(boundaries, sample):
    last_left = np.maximum(-1, boundaries[sample, 0])
    last_right = np.minimum(len(boundaries), boundaries[sample, 1])
    boundaries[last_left+1:sample+1, 1] = sample
    boundaries[sample:last_right, 0] = sample
    return boundaries


def update_sampled_boundaries_array(boundaries, left, right):
    boundaries[left+1:right, 0] = np.arange(left+1, right)
    boundaries[left+1:right, 1] = np.arange(left+1, right)


def update_progressive_sampling_info(probabilities, clusters, frame_reward, uncertainties, left, right, max_i, score):
    probabilities[clusters[left]:clusters[right]+1] = score[max_i]
    frame_reward[max_i] = np.sum(uncertainties[left:right+1])
    uncertainties[left:right+1] = 0


def label_propagation(PD, score, sampled, cluster_boundaries, sampled_boundaries, max_i, tau, oracle, probabilities, clusters, frame_reward, uncertainties, lp_enabled):

    PD[max_i] = oracle[max_i]
    score[[max_i]] = np.argmax(PD[[max_i]], axis=1)
    left, right = cluster_boundaries[max_i]
    leftT, rightT = sampled_boundaries[max_i]

    if score[max_i] == 0:
        left = adjust_boundary(left, leftT, tau, score)
        right = adjust_boundary(right, rightT, tau, score, decrease=True)

    if lp_enabled == False:
        left, right = cluster_boundaries[max_i]

    s_time = time.time()

    apply_updates(PD, score, sampled, left, right, max_i)
    sampled_boundaries = update_sampled_boundaries(sampled_boundaries, left)
    sampled_boundaries = update_sampled_boundaries(sampled_boundaries, right)
    update_sampled_boundaries_array(sampled_boundaries, left, right)
    update_progressive_sampling_info(
        probabilities, clusters, frame_reward, uncertainties, left, right, max_i, score)

    elapsed_time = time.time() - s_time
    return elapsed_time, left, right, leftT, rightT


def calculate_total_entropy(i, c_i,  probabilities, tau, score, cluster_boundaries, leftT, rightT, entropy_array):

    left, right = cluster_boundaries[i]
    if score[leftT] == 1 or (left-leftT) > tau:
        leftT = left
    if score[rightT] == 1 or (rightT-right) > tau:
        rightT = right
    entropy_sum = probabilities[c_i] * np.sum(entropy_array[left:right+1]) + (
        1-probabilities[c_i]) * np.sum(entropy_array[leftT+1:rightT])
    return entropy_sum


def update_uncertainties(range_start, range_end, uncertainties, clusters, probabilities, tau, score, boundaries, reference_start, reference_end, entropy_array):
    for index in range(range_start, range_end):
        uncertainties[index] = calculate_total_entropy(
            index, clusters[index], probabilities, tau, score, boundaries, reference_start, reference_end, entropy_array)


def calculate_boundaries(clip, IOUThreshold, total_boundaries):
    a, b = clip
    start = max(0, int(a - (b - a) * (1 / IOUThreshold - 1)))
    end = min(int(a + (b - a) * (1 - IOUThreshold)) + 1, total_boundaries - 1)
    return start, end


def calculate_indices(start, end, boundaries, cluster):
    i_values = np.arange(boundaries[start][1] + 1, boundaries[end][0])[:, None]
    cluster_range = np.arange(cluster[start] + 1, cluster[end])
    return i_values, cluster_range


def calculate_j_indices(i_values, a, b, IOUThreshold, total_boundaries):
    jl = np.where(i_values <= a, np.floor(IOUThreshold * (b - i_values + 1) + a - 1),
                  np.floor(IOUThreshold * (b - a + 1) + i_values - 1)).astype(int)
    jr = np.where(i_values <= a, np.ceil((b - a + 1) / IOUThreshold + i_values - 1),
                  np.ceil((b - i_values + 1) / IOUThreshold + a - 1)).astype(int)
    j_left = np.clip(jl, 0, total_boundaries - 1)
    j_right = np.clip(jr, 0, total_boundaries - 1)
    return j_left, j_right


def calculate_confidence(cand_clips, IOUThreshold, probabilities, cluster, boundaries):
    results = []
    total_boundaries = len(boundaries)
    for clip in cand_clips:
        start, end = calculate_boundaries(clip, IOUThreshold, total_boundaries)
        i_values, cluster_range = calculate_indices(
            start, end, boundaries, cluster)
        j_left, j_right = calculate_j_indices(
            i_values, clip[0], clip[1], IOUThreshold, total_boundaries)
        combined_prob = 0
        for i, clust_val in enumerate(cluster_range):
            cluster_j_range = np.arange(
                cluster[j_left[i]] + 1, cluster[j_right[i]])
            if cluster_j_range.size == 0:
                continue
            prob_matrix = probabilities[clust_val:cluster_j_range[-1] + 1]
            prob_products = np.cumprod(prob_matrix, axis=0)[
                cluster_j_range - clust_val]
            left_prob = 1 - probabilities[clust_val - 1]
            cluster_j_range_plus = cluster_j_range + 1
            right_prob = 1 - \
                probabilities[np.clip(
                    cluster_j_range_plus, 0, len(probabilities) - 1)]
            combined_prob += np.sum(prob_products * left_prob * right_prob)
        results.append(combined_prob)

    return np.array(results), np.mean(results)
