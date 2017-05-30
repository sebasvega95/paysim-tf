from collections import Counter
from imblearn.over_sampling import SMOTE
import logging
import numpy as np

from .data import load_data as load_normal_data


logging.basicConfig(format='%(levelname)s - %(message)s')


def apply_smote(X, y, n_jobs=8):
    sm = SMOTE(n_jobs=n_jobs)
    X_sm, y_sm = sm.fit_sample(X, y)
    return X_sm, y_sm


def load_data(filename, filename_bk, n_jobs=8):
    try:
        npzfile = np.load(filename)
        X = npzfile['features']
        y = npzfile['classname']
    except FileNotFoundError:
        logging.warning('File not found, calculating SMOTE over {} and saving in {}'.format(filename_bk, filename))
        X_normal, y_normal = load_normal_data(filename_bk)
        X, y = apply_smote(X_normal, y_normal, n_jobs=n_jobs)
        np.savez(filename, features=X, classname=y)
    return X, y

