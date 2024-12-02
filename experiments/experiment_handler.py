import os
import numpy as np
from algorithm_handler import run_algorithm
from experiment_config import DATA_DIRS, Quantile_dist
from results_io import save_results, load_results
from tools import generate_oracle_proxy, generate_cluster, generate_yolov5s, prepare_supg, findCandClips


def update_results(old_results, results):
    for key, value in results.items():
        if key not in old_results:
            old_results[key] = value
        else:
            for value, v_value in value.items():
                old_results[key][value] = v_value
    return old_results


def prepare_data_overall(V, constant, samplingRate, params):

    oracle, proxy, oracle_score, proxy_score = generate_oracle_proxy(
        os.path.join(DATA_DIRS['cdf'], f'{V}.csv'), params['op'], constant)
    cluster_threshold = str(params['clusterThreshold'])
    clusters = generate_cluster(os.path.join(
        DATA_DIRS['cluster'], V, f"{V}-{cluster_threshold}.csv"))
    ground_truth = findCandClips(oracle_score, params['op'], 0, params['tau'])
    yolo_score = generate_yolov5s(os.path.join(
        DATA_DIRS['cdf'], f'{V}.csv'), params['op'], constant)
    supg_path = prepare_supg(os.path.join(
        DATA_DIRS['supg+'], f"{V}.csv"), oracle_score, oracle, proxy)
    B = int(samplingRate * len(oracle))

    return oracle, proxy, oracle_score, proxy_score, clusters, ground_truth, yolo_score, supg_path, B


def overall_dataset(params, algorithms):

    results = {V: {name: {'Time': [], 'mIoU': [], 'Recall': [], 'Precision': []}
                   for name in algorithms.keys()} for V in params['Vs']}

    for V, constant, samplingRate in zip(params['Vs'], params['constants'], params['samplingRates']):
        oracle, proxy, oracle_score, proxy_score, clusters, ground_truth, yolo_score, supg_path, B = prepare_data_overall(
            V, constant, samplingRate, params)
        ground_truth = findCandClips(
            oracle_score, params['op'], 0, params['tau'])

        for name, function in algorithms.items():
            time_1, averageIOU, recall, precision = run_algorithm(name, function, V, ground_truth, proxy, oracle, proxy_score, oracle_score,
                                                                  B, params['op'], params['tau'], params['IOUThreshold'], params['confidence'], clusters, yolo_score, supg_path)

            results[V][name]['Time'].append(time_1)
            results[V][name]['mIoU'].append(averageIOU)
            results[V][name]['Recall'].append(recall)
            results[V][name]['Precision'].append(precision)
    old_results = load_results('overall_dataset.pkl')
    updated_results = update_results(old_results, results)
    save_results(updated_results, 'overall_dataset.pkl')
    print(updated_results)


def impact(param_name, param_values, params, algorithms):
    results = {val: {name: {'Time': [], 'mIoU': [], 'Recall': [], 'Precision': []}
                     for name in params['Vs']} for val in param_values}

    for index, val in enumerate(param_values):
        params[param_name] = val

        for V, constant, samplingRate in zip(params['Vs'], params['constants'], params['samplingRates']):
            oracle, proxy, oracle_score, proxy_score = generate_oracle_proxy(
                os.path.join(DATA_DIRS['cdf'], f"{V}.csv"), params['op'], constant)
            cluster_threshold = str(params['clusterThreshold'])
            clusters = generate_cluster(os.path.join(
                DATA_DIRS['cluster'], V, f"{V}-{cluster_threshold}.csv"))
            ground_truth = findCandClips(
                oracle_score, params['op'], 0, params['tau'])
            B = int(samplingRate * len(oracle))

            time_1, averageIOU, recall, precision = run_algorithm(
                'ARC', algorithms['ARC'], V, ground_truth, proxy, oracle, proxy_score, oracle_score, B, params['op'], params['tau'], params['IOUThreshold'], params['confidence'], clusters)

            results[val][V]['Time'].append(time_1)
            results[val][V]['mIoU'].append(averageIOU)
            results[val][V]['Recall'].append(recall)
            results[val][V]['Precision'].append(precision)
    old_results = load_results(f'{param_name}.pkl')
    updated_results = update_results(old_results, results)

    save_results(updated_results, f'{param_name}.pkl')
    print(updated_results)


def impact_opc(param_name, param_values, params, algorithms):
    results = {val[1:]: {name: {'Time': [], 'mIoU': [], 'Recall': [
    ], 'Precision': []} for name in params['Vs']} for val in param_values}

    for val in param_values:

        params['op'] = val[0]

        for V, samplingRate in zip(params['Vs'], params['samplingRates']):
            constant = Quantile_dist[V][val[1:]]
            oracle, proxy, oracle_score, proxy_score = generate_oracle_proxy(
                os.path.join(DATA_DIRS['cdf'], f"{V}.csv"), params['op'], constant)
            cluster_threshold = str(params['clusterThreshold'])
            clusters = generate_cluster(os.path.join(
                DATA_DIRS['cluster'], V, f"{V}-{cluster_threshold}.csv"))
            ground_truth = findCandClips(
                oracle_score, params['op'], 0, params['tau'])
            B = int(samplingRate * len(oracle))

            time_1, averageIOU, recall, precision = run_algorithm(
                'ARC', algorithms['ARC'], V, ground_truth, proxy, oracle, proxy_score, oracle_score, B, params['op'], params['tau'], params['IOUThreshold'], params['confidence'], clusters)

            results[val[1:]][V]['Time'].append(time_1)
            results[val[1:]][V]['mIoU'].append(averageIOU)
            results[val[1:]][V]['Recall'].append(recall)
            results[val[1:]][V]['Precision'].append(precision)
    old_results = load_results(f'{param_name}.pkl')
    updated_results = update_results(old_results, results)

    save_results(updated_results, f'{param_name}.pkl')
    print(updated_results)


def run_experiment(params, algorithms, exp_param):

    if exp_param in ['tau', 'IOUThreshold', 'confidence', 'clusterThreshold']:
        impact(exp_param, params[f'{exp_param}s'], params, algorithms)
    elif exp_param in ['constants']:
        impact_opc(exp_param, params[f'{exp_param}s'], params, algorithms)
    elif exp_param == 'overall_dataset':
        overall_dataset(params, algorithms)
    elif exp_param == 'overall_predicate':
        overall_predicate(params, algorithms)
