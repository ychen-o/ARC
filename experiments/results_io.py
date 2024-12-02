import os
import pickle

def save_results(results, filename):
    with open(os.path.join('result', filename), 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    with open(os.path.join('result', filename), 'rb') as f:
        return pickle.load(f)
