import numpy as np


def update_scores_with_oracle(proxy_score, oracle_score, indices):
    updated_score = proxy_score.copy()
    updated_score[indices] = oracle_score[indices]
    return updated_score


def entropy(prob_dist):
    return -sum(p * np.log2(p) for p in prob_dist if p > 0)


def P(score, op, constant):
    operations = {
        '>': lambda x: x > constant,
        '<': lambda x: x < constant,
        '=': lambda x: x == constant
    }
    return operations[op](score)
