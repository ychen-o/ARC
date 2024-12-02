import numpy as np
from tools import findCandClips
from score_tools import entropy,update_scores_with_oracle
from sampling_tools import select_indices_uniform,select_indices_importance

def uniform(proxy, oracle, proxy_score, oracle_score, B, op, constant, tau):
    
    indices_array = np.arange(len(proxy_score))
    selected_indices = select_indices_uniform(indices_array, B)
    updated_scores = update_scores_with_oracle(proxy_score, oracle_score, selected_indices)
    cand_clips = findCandClips(updated_scores, op, constant, tau)
    return {
        'cand_clips': cand_clips,
        'B': len(selected_indices)
    }

def importance(proxy, oracle, proxy_score, oracle_score, B, op, constant, tau):

    indices_array = np.arange(len(proxy))
    uncertainties = np.array([entropy(dist) for dist in proxy])
    selected_indices = select_indices_importance(indices_array, uncertainties, B)
    updated_scores = update_scores_with_oracle(proxy_score, oracle_score, selected_indices)
    cand_clips = findCandClips(updated_scores, op, constant, tau)
    return {
        'cand_clips': cand_clips,
        'B': len(selected_indices)
    }
