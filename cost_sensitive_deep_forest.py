
# coding: utf-8


import run_on_server as ros
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from statistics import mean
from costsensitive import WeightedOneVsRest
import sklearn.metrics as metrics
import copy
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
import multiprocessing as mp
from itertools import product


def calculate_error_sample(predictions_proba_sample, label):
    predictions_proba_wrong = copy.deepcopy(predictions_proba_sample)
    predictions_proba_wrong[label] = 1 - predictions_proba_wrong[label]
    error_sample = np.dot(predictions_proba_wrong, cost_matrix[label])
    return error_sample


def calculate_error_estimator(predictions_proba, label):
    error_all_sample = []
    for i in range(len(label)):
        error_sample = calculate_error_sample(predictions_proba[i], label[i])
        error_all_sample.append(error_sample)
    error_estimator = mean(error_all_sample)
    return error_estimator


def calculate_error_layer(predictions_proba, label):
    error_all_estimator = []
    for estimator in range(4):
        error_estimator = calculate_error_estimator(predictions_proba[estimator], label)
        error_all_estimator.append(error_estimator)
    error_layer = mean(error_all_estimator)
    return error_layer


def get_prediction(predictions_proba):
    predictions_proba_average = (predictions_proba[0] + predictions_proba[1] + 
                                       predictions_proba[2] + predictions_proba[3]) / 4
    predictions = predictions_proba_average.argmax(1)
    return predictions


# ## no cost sensitive

def deep_forest_non(estimators, X_train, X_test, y_train, y_test, cost_matrix):
    RandomForestEstimator_1 = RandomForestClassifier(n_estimators=estimators, n_jobs=-1, random_state=1)
    RandomForestEstimator_2 = RandomForestClassifier(n_estimators=estimators, n_jobs=-1, random_state=2)
    ExtraTreesEstimator_3 = ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1, random_state=3)
    ExtraTreesEstimator_4 = ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1, random_state=4)
    LayerClassifier = [RandomForestEstimator_1, RandomForestEstimator_2, 
                       ExtraTreesEstimator_3, ExtraTreesEstimator_4]
    DeepForest = []
    for layer in range(100):
        DeepForest.append(copy.deepcopy(LayerClassifier))
        if layer == 0:
            X_retrain = X_train
            X_retest = X_test
        else:
            X_retrain = np.concatenate((X_train, concatenate_predictions_proba(predictions_proba_train)), axis=1)
            X_retest = np.concatenate((X_test, concatenate_predictions_proba(predictions_proba_test)), axis=1)
        predictions_proba_train = [0, 1, 2, 3]
        predictions_proba_test = [0, 1, 2, 3]
        for estimator in range(4):
            DeepForest[layer][estimator].fit(X_retrain, y_train)
            predictions_proba_train[estimator] = DeepForest[layer][estimator].predict_proba(X_retrain)
            predictions_proba_test[estimator] = DeepForest[layer][estimator].predict_proba(X_retest)
        
        error_layer_train = calculate_error_layer(predictions_proba_train, y_train)
        error_layer_test = calculate_error_layer(predictions_proba_test, y_test)
        if (layer > 0) and (error_layer_train_ - error_layer_train < 0.01 * error_layer_train):
            break
        error_layer_train_ = error_layer_train
        predictions_layer_test = get_prediction(predictions_proba_test)
    return layer, predictions_layer_test


# ## cost sensitive


def deep_forest_cs(estimators, X_train, X_test, y_train, y_test, cost_matrix):
    CostSensitiveRandomForestEstimator_1 = WeightedOneVsRest(RandomForestClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=1))
    CostSensitiveRandomForestEstimator_2 = WeightedOneVsRest(RandomForestClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=2))
    CostSensitiveExtraTreesEstimator_3 = WeightedOneVsRest(ExtraTreesClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=3))
    CostSensitiveExtraTreesEstimator_4 = WeightedOneVsRest(ExtraTreesClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=4))
    LayerClassifier = [CostSensitiveRandomForestEstimator_1, CostSensitiveRandomForestEstimator_2, 
                       CostSensitiveExtraTreesEstimator_3, CostSensitiveExtraTreesEstimator_4]
    DeepForest = []
    C_train = np.array([cost_matrix[i] for i in y_train])
    C_test = np.array([cost_matrix[i] for i in y_test])
    for layer in range(100):
        DeepForest.append(copy.deepcopy(LayerClassifier))
        if layer == 0:
            X_retrain = X_train
            X_retest = X_test
        else:
            X_retrain = np.concatenate((X_train, concatenate_predictions_proba(predictions_proba_train)), axis=1)
            X_retest = np.concatenate((X_test, concatenate_predictions_proba(predictions_proba_test)), axis=1)
        predictions_proba_train = [0, 1, 2, 3]
        predictions_proba_test = [0, 1, 2, 3]
        for estimator in range(4):
            DeepForest[layer][estimator].fit(X_retrain, C_train)
            predictions_proba_train[estimator] = DeepForest[layer][estimator].decision_function(X_retrain)
            predictions_proba_test[estimator] = DeepForest[layer][estimator].decision_function(X_retest)
        error_layer_train = calculate_error_layer(predictions_proba_train, y_train)
        error_layer_test = calculate_error_layer(predictions_proba_test, y_test)
        if (layer > 0) and (error_layer_train_ - error_layer_train < 0.01 * error_layer_train):
            break
        error_layer_train_ = error_layer_train
        predictions_layer_test = get_prediction(predictions_proba_test)
    return layer, predictions_layer_test


def deep_forest(estimators, X_train, X_test, y_train, y_test, cost_matrix, cost_sensitive):
    if cost_sensitive:
        layer, predictions = deep_forest_cs(estimators, X_train, X_test, y_train, y_test, cost_matrix)
    else:
        layer, predictions = deep_forest_non(estimators, X_train, X_test, y_train, y_test, cost_matrix)
    accuracy, precision, recall, f1_score, cost = ros.get_metrics(y_test, predictions, cost_matrix)
    return layer, accuracy, precision, recall, f1_score, cost


# # NEW

def average(prediction_proba):
    if len(prediction_proba) == 4:
        prediction_proba_average = (prediction_proba[0] + prediction_proba[1] + 
                                    prediction_proba[2] + prediction_proba[3]) / 4
    if len(prediction_proba) == 5:
        prediction_proba_average = (prediction_proba[0] + prediction_proba[1] + 
                                    prediction_proba[2] + prediction_proba[3] + prediction_proba[3]) / 5
    return prediction_proba_average


def realign(prediction_prob, index):
    for i in range(len(prediction_prob)):
        prediction_prob[i] = pd.DataFrame(data=prediction_prob[i], index=index[i])
    df_realignment = prediction_prob[0]
    for i in range(1, len(prediction_prob)):
        df_realignment = df_realignment.append(prediction_prob[i])
    df_realignment = df_realignment.sort_index()
    return df_realignment.values


def cross_validate_estimator(clf, X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    classifers = [0, 1, 2, 3, 4]
    prediction_prob_test_cv = [0, 1, 2, 3, 4]
    prediction_prob_test = [0, 1, 2, 3, 4]
    index = [0, 1, 2, 3, 4]
    i = 0
    for train_index_cv, test_index_cv in skf.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train[train_index_cv], X_train[test_index_cv]
        y_train_cv, y_test_cv = y_train[train_index_cv], y_train[test_index_cv]
        index[i] = test_index_cv
        classifers[i] = copy.deepcopy(clf)
        if cost_sensitive:
            C = np.array([cost_matrix[i] for i in y_train_cv])
            classifers[i].fit(X_train_cv, C)
            prediction_prob_test_cv[i] = classifers[i].decision_function(X_test_cv)
            prediction_prob_test[i] = classifers[i].decision_function(X_test)
        else:
            classifers[i].fit(X_train_cv, y_train_cv)
            prediction_prob_test_cv[i] = classifers[i].predict_proba(X_test_cv)
            prediction_prob_test[i] = classifers[i].predict_proba(X_test)
        i += 1
    prediction_prob_train = realign(prediction_prob_test_cv, index)    
    prediction_prob_test = average(prediction_prob_test)
    return prediction_prob_train, prediction_prob_test



def layer_estimate(estimators, X_retrain, y_train, X_retest, y_test, cost_matrix, cost_sensitive):
    non_forests = [RandomForestClassifier(n_estimators=estimators, n_jobs=-1), 
                   RandomForestClassifier(n_estimators=estimators, n_jobs=-1), 
                   ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1), 
                   ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1)]
    cs_forests = [WeightedOneVsRest(RandomForestClassifier(n_estimators=estimators, n_jobs=-1)), 
                  WeightedOneVsRest(RandomForestClassifier(n_estimators=estimators, n_jobs=-1)), 
                  WeightedOneVsRest(ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1)), 
                  WeightedOneVsRest(ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1))]
    if cost_sensitive:
        forests = cs_forests
    else:
        forests = non_forests
    prediction_prob_train = [0, 1, 2, 3]
    prediction_prob_test = [0, 1, 2, 3]
    i = 0
    for frs in forests:
        prediction_prob_train[i], prediction_prob_test[i] = cross_validate_estimator(
            frs, X_retrain, y_train, X_retest, y_test, cost_matrix, cost_sensitive)
        i += 1
    return prediction_prob_train, prediction_prob_test



def get_layer_metrics(prediction_prob_train, prediction_prob_test, y_train, y_test, cost_matrix):
    prediction_train = average(prediction_prob_train)
    prediction_test = average(prediction_prob_test)
    predictions_train = prediction_train.argmax(1)
    predictions_test = prediction_test.argmax(1)
    accuracy_train, precision_train, recall_train, f1_score_train, cost_train = ros.get_metrics(
        y_train, predictions_train, cost_matrix)
    accuracy_test, precision_test, recall_test, f1_score_test, cost_test = ros.get_metrics(
        y_test, predictions_test, cost_matrix)
    return [accuracy_train, precision_train, recall_train, f1_score_train, cost_train, 
            accuracy_test, precision_test, recall_test, f1_score_test, cost_test]



def concatenate_predictions_proba(predictions_proba):
    predictions_proba_all = np.concatenate((predictions_proba[0], predictions_proba[1], 
                                            predictions_proba[2], predictions_proba[3]), axis=1)
    return predictions_proba_all


def train_test_once(estimators, X_train, X_test, y_train, y_test, cost_matrix, cost_sensitive):
    stopping = 0
    for layer in range(100):
        if layer == 0:
            X_retrain = X_train
            X_retest = X_test
        else:
            X_retrain = np.concatenate((X_train, concatenate_predictions_proba(prediction_prob_train)), axis=1)
            X_retest = np.concatenate((X_test, concatenate_predictions_proba(prediction_prob_test)), axis=1)
        prediction_prob_train, prediction_prob_test = layer_estimate(
            estimators, X_retrain, y_train, X_retest, y_test, cost_matrix, cost_sensitive)
        [accuracy_train, precision_train, recall_train, f1_score_train, cost_train, 
         accuracy_test, precision_test, recall_test, f1_score_test, cost_test] = get_layer_metrics(
            prediction_prob_train, prediction_prob_test, y_train, y_test, cost_matrix)
        if cost_sensitive:
            if (layer == 0) or (best_cost_train - cost_train > 0.001 * best_cost_train):
                best_layer = copy.deepcopy(layer) + 1
                best_accuracy_train = copy.deepcopy(accuracy_train)
                best_cost_train = copy.deepcopy(cost_train)
                best_accuracy_test = copy.deepcopy(accuracy_test)
                best_precision_test = copy.deepcopy(precision_test)
                best_recall_test = copy.deepcopy(recall_test)
                best_f1_score_test = copy.deepcopy(f1_score_test)
                best_cost_test = copy.deepcopy(cost_test)
                stopping = 0
            else:
                stopping += 1
        else:
            if (layer == 0) or (accuracy_train - best_accuracy_train > 0.001 * best_accuracy_train):
                best_layer = copy.deepcopy(layer) + 1
                best_accuracy_train = copy.deepcopy(accuracy_train)
                best_cost_train = copy.deepcopy(cost_train)
                best_accuracy_test = copy.deepcopy(accuracy_test)
                best_precision_test = copy.deepcopy(precision_test)
                best_recall_test = copy.deepcopy(recall_test)
                best_f1_score_test = copy.deepcopy(f1_score_test)
                best_cost_test = copy.deepcopy(cost_test)
                stopping = 0
            else:
                stopping += 1
        if stopping == 3:
            break
    return best_layer, best_accuracy_test, best_precision_test, best_recall_test, best_f1_score_test, best_cost_test



def get_mean_results(estimators, X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, time):

    layer_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    cost_list = []

    for i in range(time):  # time = repeat time for experiments
        layer_, accuracy_, precision_, recall_, f1_score_, cost_ = train_test_once(
            estimators, X_train, X_test, y_train, y_test, cost_matrix, cost_sensitive)

        layer_list.append(layer_)
        accuracy_list.append(accuracy_)
        precision_list.append(precision_)
        recall_list.append(recall_)
        f1_score_list.append(f1_score_)
        cost_list.append(cost_)

    layer = mean(layer_list)
    accuracy = mean(accuracy_list)
    precision = mean(precision_list)
    recall = mean(recall_list)
    f1_score = mean(f1_score_list)
    cost = mean(cost_list)

    return layer, accuracy, precision, recall, f1_score, cost


def train_test_df(data, price, smallest_classes, classes, discretization, cost_sensitive, estimators, time):
    df_label = ros.choose_discretization(price, classes, discretization, smallest_classes)
    X_train, X_test, y_train, y_test, cost_matrix = ros.prepare_data(data, df_label, classes)
    layer, accuracy, precision, recall, f1_score, cost = get_mean_results(
        estimators, X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, time)
    df_save = pd.DataFrame([[classes, discretization, cost_sensitive, estimators, time, layer, accuracy, precision, recall, f1_score, cost]])
    df_save.to_csv(os.path.abspath('..') + '\\results\\results_df.csv',
                   mode='a', encoding='utf-8', index=False, header=False)
    print(classes, discretization, cost_sensitive, estimators, time, layer, accuracy, precision, recall, f1_score, cost)


if __name__ == '__main__':

    data, price = ros.input_row_data()
    estimators = 10
    time = 1
    smallest_classes = 4
    biggest_classes = 20
    classes = list(np.arange(smallest_classes, biggest_classes))
    discretization = ['EPI', 'EOH', 'KM', 'KMI', 'KMN', 'KMD']
    cost_sensitive = [False, True]
    pool = mp.Pool(processes=10, maxtasksperchild=1)
    pool.starmap(train_test_df, product(
        [data], [price], [smallest_classes], classes, discretization, cost_sensitive, [estimators], [time]))
