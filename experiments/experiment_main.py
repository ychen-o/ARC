from experiment_handler import run_experiment
from algorithm_handler import uniform,importance,arc
import multiprocessing

import os

algorithms_dataset =  {
            'Oracle-Only':None,
            'YOLOv5s-Only':None,
            'CMDN-Uniform': uniform,
           'CMDN-Importance': importance,
            'SUPGrt+':None,
            'SUPGpt+':None,
            'ARC': arc  
}

algorithms_impact =  {
            'ARC': arc  
}

parameter_sets = {
        'overall_dataset': {
            'op': '>',   
            'tau': 300,
            'confidence':0.9,
            'IOUThreshold':0.9,    
            'samplingRates': [0.1,0.1,0.1,0.1,0.1],
            'constants':[3,3,10,5,10],
            'Vs': ['jackson-town-square','amsterdam','taipei-hires','venice-rialto','venice-grand-canal'],
            'clusterThreshold':0.001
        },
    
        'impact': {
            'op': '>',
            'constants': [3,3,10,5,10],
            'constantss': ['>P2.5','>P5','>P10','>P15','>P20'],
            'tau':300, 
            'taus': [180,210,240,270,300,330,360,390,420],
            'IOUThreshold':0.9,     
            'IOUThresholds': [0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.95,0.975],    
            'confidence':0.9,
            'confidences':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
            'clusterThreshold':0.001,
            'clusterThresholds': [0.00025,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004],
            'samplingRates': [0.1,0.1,0.1,0.1,0.1],
            'Vs': ['jackson-town-square','amsterdam','taipei-hires','venice-rialto','venice-grand-canal'], 
        },
}
def main():
    
    if not os.path.exists('result'):
        os.makedirs('result')

    # Cost decomposition experiments can be performed by uncommenting part of the calculate_time function in metrics.py, while ablation experiments can be performed by modifying the default parameters of the arc function in arc.py (tc_enabled, ps_enabled, and lp_enabled).
    
    tasks = [
        (run_experiment, (parameter_sets.get('overall_dataset', {}), algorithms_dataset,'overall_dataset') ),
        #(run_experiment, (parameter_sets.get('impact', {}), algorithms_impact, 'tau') ),
        #(run_experiment, (parameter_sets.get('impact', {}), algorithms_impact, 'IOUThreshold') ),
        #(run_experiment, (parameter_sets.get('impact', {}), algorithms_impact, 'confidence') ),
        #(run_experiment, (parameter_sets.get('impact', {}), algorithms_impact, 'clusterThreshold') ),
        #(run_experiment, (parameter_sets.get('impact', {}), algorithms_impact, 'constants') ),
    ]

    for func, args in tasks:
        func(*args)  
        print(f"Completed {func.__name__} with arguments {args}")
    

main()