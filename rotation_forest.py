#!/usr/bin/env python
# coding: utf-8

import run_on_server as ros
import pandas as pd
import numpy as np
import os
from statistics import mean
from rotation_forest import RotationForestClassifier
import multiprocessing as mp
from itertools import product


def train_test_once(X_train, y_train, X_test, y_test, cost_matrix):
    rof = RotationForestClassifier()
    rof.fit(X_train, y_train)
    predictions = rof.predict(X_test)
    # use same function with cost-sensitive
    return predictions

def get_mean_results(X_train, y_train, X_test, y_test, cost_matrix, classes, time):

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    cost_list = []

    for i in range(time):  # time = repeat time for experiments
        predictions = train_test_once(X_train, y_train, X_test, y_test, cost_matrix)
        accuracy_, precision_, recall_, f1_score_, cost_ = ros.get_metrics(y_test, predictions, cost_matrix)

        accuracy_list.append(accuracy_)
        precision_list.append(precision_)
        recall_list.append(recall_)
        f1_score_list.append(f1_score_)
        cost_list.append(cost_)

    accuracy = mean(accuracy_list)
    precision = mean(precision_list)
    recall = mean(recall_list)
    f1_score = mean(f1_score_list)
    cost = mean(cost_list)

    return accuracy, precision, recall, f1_score, cost


def rotation_forest(data, price, smallest_classes, classes, discretization, time):
    df_label = ros.choose_discretization(price, classes, discretization, smallest_classes)
    X_train, X_test, y_train, y_test, cost_matrix = ros.prepare_data(data, df_label, classes)
    accuracy, precision, recall, f1_score, cost = get_mean_results(X_train, y_train, X_test, y_test, cost_matrix, classes, time)
    df_save = pd.DataFrame([[classes, discretization, time,
                             accuracy, precision, recall, f1_score, cost]])
    df_save.to_csv(os.path.abspath('..') + '\\results\\results_rof.csv',
                   mode='a', encoding='utf-8', index=False, header=False)
    print(df_save)



if __name__ == '__main__':
    data, price = ros.input_row_data()
    estimators = 10
    time = 1
    smallest_classes = 4
    biggest_classes = 20
    classes = list(np.arange(smallest_classes, biggest_classes))
    discretization = ['EPI', 'EOH', 'KM', 'KMI', 'KMN', 'KMD']
    pool = mp.Pool(processes=3, maxtasksperchild=1)
    pool.starmap(rotation_forest, product(
        [data], [price], [smallest_classes], classes, discretization, [time]))
