import os

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIRS = {
    'cdf': os.path.join(BASE_DIR, 'data', 'CDF'),
    'cluster': os.path.join(BASE_DIR, 'data', 'cluster'),
    'supg+': os.path.join(BASE_DIR, 'data', 'SUPG+'),
}

# Precomputed to estimate end-to-end overhead
TIME_CONFIG = {
    "oracle_time": {
        'amsterdam': 0.0860, 'venice-rialto': 0.0897, 'venice-grand-canal': 0.0874,
        'taipei-hires': 0.0878, 'jackson-town-square': 0.0711
    },
    "proxy_time": {
        'amsterdam': 0.00123762, 'venice-rialto': 0.00087032, 'venice-grand-canal': 0.00057471,
        'taipei-hires': 0.00124378, 'jackson-town-square': 0.00078616
    },
    "YOLOv5s_time": {
        'amsterdam': 0.014298, 'venice-rialto': 0.0222, 'venice-grand-canal': 0.0151,
        'taipei-hires': 0.01430001, 'jackson-town-square': 0.0149
    },
    "cluster_time": {
        'amsterdam': 30.6029, 'venice-rialto': 22.4119, 'venice-grand-canal': 34.5492,
        'taipei-hires': 32.2784, 'jackson-town-square': 25.1041
    },

}

Quantile_dist = {'jackson-town-square': {'P2.5': 4, 'P5': 3, 'P10': 3, 'P15': 2, 'P20': 2, 'P25': 2}, 'amsterdam': {'P2.5': 3, 'P5': 3, 'P10': 3, 'P15': 2, 'P20': 2, 'P25': 2}, 'taipei-hires': {'P2.5': 12, 'P5': 11,
                                                                                                                                                                                                  'P10': 10, 'P15': 9, 'P20': 8, 'P25': 8}, 'venice-rialto': {'P2.5': 6, 'P5': 6, 'P10': 5, 'P15': 5, 'P20': 4, 'P25': 4}, 'venice-grand-canal': {'P2.5': 11, 'P5': 11, 'P10': 10, 'P15': 9, 'P20': 9, 'P25': 9}}
