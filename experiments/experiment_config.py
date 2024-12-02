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
