import numpy as np

def select_indices_uniform(indices, B):
    return np.random.choice(indices, B, replace=False)

def select_indices_importance(indices, uncertainties, B):
    
    if uncertainties.sum() == 0:
        return None
    else:
        probabilities = uncertainties / uncertainties.sum()  
    non_zero_prob_count = np.sum(probabilities > 0)  
    adjusted_B = min(B, non_zero_prob_count)  
    
    return np.random.choice(indices, adjusted_B, replace=False, p=probabilities)
