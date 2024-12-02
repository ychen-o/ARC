import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import jensenshannon
from arc_config import DATA_DIRS

def init_probabilities(PD, index, values):
    PD[PD[:, index] == 1] = values

def build_cluster_boundaries(clusters):  
    clusters = np.array(clusters)
    change_points = np.where(clusters[:-1] != clusters[1:])[0] + 1
    change_points = np.concatenate(([0], change_points, [len(clusters)]))
    boundaries = np.zeros((len(clusters), 2), dtype=int)
    boundaries[:, 0] = np.repeat(change_points[:-1], np.diff(change_points))
    boundaries[:, 1] = np.repeat(change_points[1:] - 1, np.diff(change_points))
    return boundaries

def init_cluster_uncertainties(clusters, entropy_array, proxy):
    boundaries = build_cluster_boundaries(clusters)
    unique_clusters = np.unique(clusters)
    uncertainties = np.zeros(len(clusters))
    probabilities = np.zeros(len(unique_clusters))
    for i in range(len(uncertainties)):
        left, right = boundaries[i]
        uncertainties[i] = np.sum(entropy_array[left:right + 1])
        probabilities[clusters[left]] = np.mean(proxy[left:right + 1, 1]) 
    return boundaries, uncertainties, probabilities

def js_divergence(distribution1, distribution2):
    return jensenshannon(distribution1, distribution2)
    
def cluster_load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['predicates', 'yolov5s'])
    distributions = np.diff(data.to_numpy(), axis=1, prepend=0)
    return distributions

def compute_joint_distribution(distribution1, distribution2):
    n_rows = distribution1.shape[0]
    joint_distributions = [np.outer(distribution1[i], distribution2[i]).flatten() for i in range(n_rows)]
    return np.array(joint_distributions)

def save_clusters(labels_df, file_path):
    labels_df.to_csv(file_path, index=False)
    return labels_df

def load_clusters(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def perform_clustering(distributions, threshold, directory, cluster_folder):

    output_folder = os.path.join(cluster_folder, directory)
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f'{directory}-{threshold}.csv')
    
    if os.path.exists(file_path):
        print(f"Loading existing cluster file for {directory} with threshold {threshold}")
        return load_clusters(file_path)
    
    labels = np.zeros(len(distributions))
    current_label = 0
    p_a = distributions[0]

    for i in range(1, len(distributions)):
        p_b = distributions[i]
        if js_divergence(p_a, p_b) > threshold:
            p_a = p_b
            current_label += 1
        labels[i] = current_label

    labels_df = pd.DataFrame(labels, columns=['label'])
    return save_clusters(labels_df, file_path)

def cluster(directorys, thress, cdf_folder, cluster_folder, return_labels=False):
    for directory in directorys:
        for thres in thress:
            file_path = os.path.join(cdf_folder, f'{directory}.csv')
            distributions = cluster_load_data(file_path)
            labels_df = perform_clustering(distributions, thres, directory, cluster_folder)
            if return_labels:
                return labels_df
            
def cluster4join(directorys1, directorys2, thress, cdf_folder, cluster_folder, return_labels=False):
    for directory1, directory2 in zip(directorys1, directorys2):
        for thres in thress:
            file_path1 = os.path.join(cdf_folder, f'{directory1}.csv')
            file_path2 = os.path.join(cdf_folder, f'{directory2}.csv')
            distribution1 = cluster_load_data(file_path1)
            distribution2 = cluster_load_data(file_path2)
            joint_distributions = compute_joint_distribution(distribution1, distribution2)
            perform_clustering(joint_distributions, thres, f'{directory1}-{directory2}', cluster_folder)
            if return_labels:
                return labels_df

#directorys = ['venice-rialto','venice-grand-canal','taipei-hires','amsterdam','jackson-town-square']
#thress = [0.0015,0.0025,0.003,0.0035]
#cluster(directorys, thress, DATA_DIRS['cdf'], DATA_DIRS['cluster'])

