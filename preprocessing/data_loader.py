import numpy as np
import os
from preprocessing_config import DATA_DIRS


def load_data(directory, file_name):
    return np.load(os.path.join(DATA_DIRS[directory], file_name))


def load_model_results(directory, file_name):
    return load_data(directory, file_name + '.npy')
