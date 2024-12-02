import numpy as np
import pandas as pd
from score_tools import P

def generate_oracle_proxy(path, op, constant):
    df = pd.read_csv(path)
    predicates = df['predicates'].values
    o_1 = np.vectorize(lambda x: P(x, op, constant))(predicates)
    o_0 = 1 - o_1
    oracle = np.column_stack((o_0, o_1))
    p_name= str(constant)
    if op == '>':
        p_0 = df[p_name].values
        p_1 = 1 - p_0
        proxy = np.column_stack((p_0, p_1))
    proxy_score = np.argmax(proxy, axis=1)
    oracle_score = np.argmax(oracle, axis=1)

    return oracle, proxy,oracle_score,proxy_score

def generate_cluster(path):
    df = pd.read_csv(path)
    return df['label'].values.astype(int)

def generate_yolov5s(path,op,constant):
    df = pd.read_csv(path)
    predicates = df['yolov5s'].values
    o_1 = np.vectorize(lambda x: P(x, op, constant))(predicates)
    o_0 = 1 - o_1
    oracle = np.column_stack((o_0, o_1))
    oracle_score = np.argmax(oracle, axis=1)
    return oracle_score

def prepare_supg(path,oracle_score,oracle,proxy):
    data = {
        'id': np.arange(len(oracle)),
        'label': oracle_score == 1,
        'proxy_score': proxy[:, 1]
    }
    df_output = pd.DataFrame(data)
    df_output.to_csv(path, index=False)
    return path
    
def findCandClips(score, op, constant, tau): 
    conditions = P(score, op, constant)
    diffs = np.diff(conditions.astype(int))
    start_indices = np.where(diffs == 1)[0] + 1
    end_indices = np.where(diffs == -1)[0]  
    if conditions[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if conditions[-1]:
        end_indices = np.append(end_indices, len(conditions) - 1)
    valid_mask = (end_indices - start_indices + 1) >= tau
    valid_start_indices = start_indices[valid_mask]
    valid_end_indices = end_indices[valid_mask]
    
    return np.column_stack((valid_start_indices, valid_end_indices)) 

def calculate_start(a, b, IOUThreshold, tau):
    extend = (1 / IOUThreshold - 1) * (b - a)
    start_adjusted = a - extend
    return max(0, int(min(start_adjusted, a - tau)))

def calculate_end(a, b, IOUThreshold, tau, n):
    extend = (1 / IOUThreshold - 1) * (b - a)
    end_adjusted = b + extend
    return min(n - 1, int(max(end_adjusted, b + tau)))

def calculate_reliability(observed_matches, total_observations):
    return observed_matches / total_observations if total_observations > 0 else 0