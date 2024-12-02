import os

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIRS = {
    'cdf': os.path.join(BASE_DIR, 'data', 'CDF'),
    'cluster': os.path.join(BASE_DIR, 'data', 'cluster'),
}
