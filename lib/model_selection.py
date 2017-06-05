from hostlist import expand_hostlist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import itertools
import numpy as np
import os
import tensorflow as tf


def cross_validation(estimator, X, y, n_splits=3, server=''):
    kf = KFold(n_splits=n_splits)
    num_features = X.shape[1]
    train_scores = []
    test_scores = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        clf = estimator(num_features, server=server)
        clf.fit(X_train, y_train)
        predictions_train = clf.predict(X_train)
        predictions_test = clf.predict(X_test)
        #clf.close()
        accuracy_train = accuracy_score(y_train, predictions_train)
        accuracy_test = accuracy_score(y_test, predictions_test)
        train_scores.append(accuracy_train)
        test_scores.append(accuracy_test)
    return train_scores, test_scores


def setup_slurm_cluster(port=27856, job_name='model_selection'):
    tf_hostlist = ['{}:{}'.format(host, port)
                   for host in expand_hostlist(os.environ['SLURM_NODELIST'])]
    cluster = tf.train.ClusterSpec({job_name: tf_hostlist})
    return cluster


def calc_work(param_grid, n_tasks):
    workload = {i : [] for i in range(n_tasks)}
    param_grid = ParameterGrid(param_grid)
    for idx, params in zip(itertools.cycle(range(n_tasks)), param_grid):
        workload[idx].append(params)
    return workload


def grid_search(estimator, X, y, param_grid, port=27856, job_name='model_selection', n_splits=3):
    cluster = setup_slurm_cluster(port=27856, job_name='model_selection')
    task_index = int(os.environ['SLURM_PROCID'])
    n_tasks = int(os.environ['SLURM_NPROCS'])
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    max_scores = {}
    for i in range(n_tasks):
        max_scores[i] = {'params': {}, 'score': -float('inf')}

    workload = calc_work(param_grid, n_tasks)
    print(workload)
    with tf.device(tf.train.replica_device_setter(
        worker_device='/job:{}/task:{}'.format(job_name, task_index),
        cluster=cluster)):
        for params in workload[task_index]:
            degree = params['poly_degree']
            lamda = params['lambda']
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_cv = poly.fit_transform(X)
            scaler = StandardScaler()
            X_cv = scaler.fit_transform(X_cv)
            y_cv = y
            print('-->', task_index, X_cv.shape)
            _, test_scores = cross_validation(estimator, X_cv, y_cv, n_splits=n_splits, server=server.target)
            cv_score = np.mean(test_scores)
            print('Task', task_index)
            print('  Params =', params)
            print('  Score  =', cv_score)
            if cv_score > max_scores[task_index]['score']:
                max_scores[task_index]['score'] = cv_score
                max_scores[task_index]['params'] = params
    return max_scores

