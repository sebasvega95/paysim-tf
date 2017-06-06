from datetime import timedelta
from hostlist import expand_hostlist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from time import time
import itertools
import numpy as np
import os
import zmq

from lib.estimator import BinaryLogisticRegression
from lib.model_selection import grid_search
from lib.smote_data import load_data


def calc_work(param_grid, n_tasks):
    workload = {i : [] for i in range(n_tasks)}
    param_grid = ParameterGrid(param_grid)
    for idx, params in zip(itertools.cycle(range(1, n_tasks)), param_grid):
        workload[idx].append(params)
    return workload


def test_final_clf(estimator, weights, bias, X, y, params):
    degree = params['poly_degree']
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    num_features = X.shape[1]
    clf = estimator(num_features, weights, bias)
    predictions = clf.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy


def main():
    port = 27698
    task_index = int(os.environ['SLURM_PROCID'])
    n_tasks = int(os.environ['SLURM_NPROCS'])
    hostlist = ['{}:{}'.format(host, port)
                for host in expand_hostlist(os.environ['SLURM_NODELIST'])]

    context = zmq.Context()
    receive_socket = context.socket(zmq.PULL)
    receive_socket.bind('tcp://*:{}'.format(port))

    start = time()
    print('Loading data...')
    X, y = load_data('smote_data.npz', 'data.csv')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    if task_index == 0: # PS
        param_grid = {
           'lambda': [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
           'poly_degree': [2, 3, 4, 5]
        }
        workload = calc_work(param_grid, n_tasks)
        for i, node in enumerate(hostlist[1:]):
            worker_socket = context.socket(zmq.PUSH)
            worker_socket.connect('tcp://{}'.format(node))
            message = {'params' : workload[i + 1]}
            worker_socket.send_json(message)

        max_score = -float('inf')
        for i in range(n_tasks - 1):
            work_ans = receive_socket.recv_json()
            print('Task index', work_ans['task_index'])
            print('    Max CV score', work_ans['best_params']['score'])
            print('    Using', work_ans['best_params']['params'])
            if work_ans['best_params']['score'] > max_score:
                max_score = work_ans['best_params']['score']
                params = work_ans['best_params']['params']
                weights = work_ans['best_params']['weights']
                bias = work_ans['best_params']['bias']
        end = time()
        print('All tasks done in', timedelta(seconds=end - start))
        print('Picked params', params)
        test_accuracy = test_final_clf(BinaryLogisticRegression, weights, bias, X_test, y_test, params)
        print('Test accuracy:', test_accuracy)
    else: # Worker
        workload = receive_socket.recv_json()
        ps_socket = context.socket(zmq.PUSH)
        ps_socket.connect('tcp://{}'.format(hostlist[0]))

        best_params = grid_search(BinaryLogisticRegression, X_train, y_train, workload['params'], n_splits=10)
        work_ans = {'task_index': task_index, 'best_params': best_params}
        ps_socket.send_json(work_ans)
        end = time()
        print('Task', task_index, 'done in', timedelta(seconds=end - start))


if __name__ == '__main__':
    main()

