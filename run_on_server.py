#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from statistics import mean
import multiprocessing as mp
import sklearn
from itertools import product, repeat
from random import choice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from costsensitive import WeightedOneVsRest



def price_boxplot(price, whis=3):
    plt.figure(figsize=[17,1])
    plt.boxplot(price, vert=False, whis=whis) # 重度异常点, whis=default 为中度异常点
    plt.show()


def input_row_data():
    df = pd.read_csv('data.csv')
    data = df[['rents', 'response', 'age', 'mile', 'displacement', 'extra_fee', 'seats', 'gps', 'mp3', 'f2f', 'steer',
               'recommendation', 'automatic_order', 'driving_experience', 'out_of_town', 'weekday',
               'city_beijing', 'city_guangzhou', 'gender_female', 'gender_male', 'area_asia', 'area_china',
               'area_america', 'area_germany']]
    price = df['price']
    return data, price

def get_discretization_results(df_label, classes):
    df_discretization = pd.DataFrame(columns=[
        'class', 'minimum', 'maximum', 'mean', 'number', 'interval', 'density'], index=range(classes))
    for i in range(classes):
        df_discretization.loc[i]['class'] = i
        df_sublabel = df_label['price'][df_label['label'] == i]
        df_discretization.loc[i]['minimum'] = df_sublabel.min()
        df_discretization.loc[i]['maximum'] = df_sublabel.max()
        df_discretization.loc[i]['mean'] = df_sublabel.mean()
        df_discretization.loc[i]['number'] = len(df_sublabel)
        df_discretization.loc[i]['interval'] = (
            df_discretization.loc[i]['maximum'] - df_discretization.loc[i]['minimum'])
        if df_discretization.loc[i]['interval']:
            df_discretization.loc[i]['density'] = (
                df_discretization.loc[i]['number'] / df_discretization.loc[i]['interval'])
        else:
            df_discretization.loc[i]['density'] = df_discretization.loc[i]['number']
    df_discretization.sort_values("mean",inplace=True)
    return df_discretization


def find_max_interval(df_discretization):
    df_max = df_discretization['interval'].max()
    max_interval = np.random.choice([df_max])
    max_class = int(np.random.choice(df_discretization['class'][df_discretization['interval']==max_interval].values))
    max_class = np.random.choice([max_class])
    max_minimum = int(df_discretization['minimum'][df_discretization['class']==max_class].values)
    max_maximum = int(df_discretization['maximum'][df_discretization['class']==max_class].values)
    return max_interval, max_minimum, max_maximum, max_class


def find_min_number(df_discretization):
    df_min = df_discretization['number'].min()
    min_number = np.random.choice([df_min])
    min_class = int(np.random.choice(df_discretization['class'][df_discretization['number']==min_number].values))
    min_minimum = int(df_discretization['minimum'][df_discretization['class']==min_class].values)
    min_maximum = int(df_discretization['maximum'][df_discretization['class']==min_class].values)
    return min_number, min_minimum, min_maximum, min_class


def find_min_density(df_discretization):
    df_min = df_discretization['density'].min()
    min_density = np.random.choice([df_min])
    min_class = int(np.random.choice(df_discretization['class'][df_discretization['density']==min_density].values))
    min_class = np.random.choice([min_class])
    min_minimum = int(df_discretization['minimum'][df_discretization['class']==min_class].values)
    min_maximum = int(df_discretization['maximum'][df_discretization['class']==min_class].values)
    return min_density, min_minimum, min_maximum, min_class


def equally_probable_intervals(price, classes):
    amount = len(price)/classes # amount in every interval
    c = 0
    label = []
    for n in range(len(price)): # same price may be classfied into two adjacent intervals
        label.append(c)
        if n > int((c+1)*amount):
            c += 1
            
    label = pd.Series(data=label, name='label')
    df_label = pd.concat([price, label], axis=1)

    for i in range(1, classes):
        # ensure that same price is in the bigger class, 
        # because interval with same price may occurs in bigger class
        same_price = df_label['price'][df_label['label']==i].iloc[0]
        df_label['label'][df_label['price']==same_price] = i

    return df_label


def equally_number_intervals(price, classes): # exactly equal
    ## used for k-means, because this discretization is only used to find centroid for k-means
    amount = len(price)/classes # amount in every interval
    c = 0
    label = []
    for n in range(len(price)): # same price may be classfied into two adjacent intervals
        label.append(c)
        if n > int((c+1)*amount):
            c += 1
    
    label = pd.Series(data=label, name='label')
    df_label = pd.concat([price, label], axis=1)
    return df_label


# ### every one hundred

def every_one_hundred(price, classes):

    interval = list(range(0, classes*100, 100))
    interval.append(max(price)+1)

    label = pd.cut(price, interval, labels=list(range(classes)))
    label = label.astype('int')
    # reform the 'label' from 'category' to 'int'
    # sklearn only accept 'int' as 'label', though int is category infact

    label = pd.Series(data=label, name='label')
    df_label = pd.concat([price, label], axis=1)

    return df_label


def find_centroid(df_label, classes):
    
    centroid = []
    for i in range(classes):
        df_sub = df_label['price'][df_label['label'] == i]
        mean = df_sub.mean()
        centroid.append(mean)
        
    return centroid

def k_means(price, classes):
    
    df_label = equally_number_intervals(price, classes)
    centroid = find_centroid(df_label, classes)
    kmeans = KMeans(n_clusters=classes, init=np.array(centroid).reshape(-1, 1)).fit(
        price.values.reshape(-1, 1))

    label = pd.Series(data=kmeans.labels_, name='label')
    df_label = pd.concat([price, label], axis=1)

    return df_label

def go_to_median(df_label, df_discretization, modified_minimun, modified_maximum, modified_class):
    median = int(df_label['price'].median())
    if modified_minimun > median:
        closer_maximum = df_discretization['maximum'][df_discretization['class']==modified_class-1].values
        df_label['price'][df_label['label']==modified_class] = closer_maximum
    elif modified_maximum < median:
        closer_minimun = df_discretization['minimum'][df_discretization['class']==modified_class+1].values
        df_label['price'][df_label['label']==modified_class] = closer_minimun
    return df_label


def resort_label(df_label, classes):
    df_discretization = get_discretization_results(df_label, classes)
    class_list_before = list(df_discretization['class'])
    class_list_after = list(df_discretization.sort_values(['mean'])['class'])
    df_label['label'].replace(class_list_before, class_list_after, inplace=True)
    return df_label


def k_means_interval(price, classes, smallest_classes):
    df_label = k_means(price, smallest_classes)
    
    while 1: # repeat all the time
        df_label = k_means(df_label['price'], smallest_classes)
        df_discretization = get_discretization_results(df_label, smallest_classes)
        last_max_interval, last_max_minimum, last_max_maximum, last_max_class = find_max_interval(
            df_discretization)
        non_break = True
        
        for classes_ in range(smallest_classes+1, classes+1):
            df_label = k_means(df_label['price'], classes_)
            df_discretization = get_discretization_results(df_label, classes_)
            max_interval, max_minimum, max_maximum, max_class = find_max_interval(df_discretization)
            if last_max_minimum == max_minimum and last_max_maximum == max_maximum:
                df_label = go_to_median(
                    df_label, df_discretization, max_minimum, max_maximum, max_class)
                non_break = False
                break
            else:
                last_max_minimum = copy.deepcopy(max_minimum)
                last_max_maximum = copy.deepcopy(max_maximum)
        if non_break:
            break

    df_label = resort_label(df_label, classes)
    return df_label


def k_means_number(price, classes, smallest_classes):
    df_label = k_means(price, smallest_classes)
    
    while 1: # repeat all the time
        df_label = k_means(df_label['price'], smallest_classes)
        df_discretization = get_discretization_results(df_label, smallest_classes)
        last_min_number, last_min_minimum, last_min_maximum, last_min_class = find_min_number(
            df_discretization)
        non_break = True
        
        for classes_ in range(smallest_classes+1, classes+1):
            df_label = k_means(df_label['price'], classes_)
            df_discretization = get_discretization_results(df_label, classes_)
            min_number, min_minimum, min_maximum, min_class = find_min_number(df_discretization)
            if last_min_minimum == min_minimum and last_min_maximum == min_maximum:
                df_label = go_to_median(
                    df_label, df_discretization, min_minimum, min_maximum, min_class)
                non_break = False
                break
            else:
                last_min_minimum = copy.deepcopy(min_minimum)
                last_min_maximum = copy.deepcopy(min_maximum)

        if non_break:
            break

    df_label = resort_label(df_label, classes)
    return df_label



def k_means_density(price, classes, smallest_classes):
    df_label = k_means(price, smallest_classes)

    while 1: # repeat all the time
        df_label = k_means(df_label['price'], smallest_classes)
        df_discretization = get_discretization_results(df_label, smallest_classes)
        last_min_density, last_min_minimum, last_min_maximum, last_min_class = find_min_density(
            df_discretization)
        non_break = True
        
        for classes_ in range(smallest_classes+1, classes+1):
            df_label = k_means(df_label['price'], classes_)
            df_discretization = get_discretization_results(df_label, classes_)
            min_density, min_minimum, min_maximum, min_class = find_min_density(df_discretization)
            if last_min_minimum == min_minimum and last_min_maximum == min_maximum:
                df_label = go_to_median(
                    df_label, df_discretization, min_minimum, min_maximum, min_class)
                non_break = False
                break
            else:
                last_min_minimum = copy.deepcopy(min_minimum)
                last_min_maximum = copy.deepcopy(min_maximum)

        if non_break:
            break

    df_label = resort_label(df_label, classes)
    
    return df_label


# ### choose discretization



def choose_discretization(price, classes, discretization, smallest_classes):
    if discretization == 'EPI': # 'equally_probable_intervals'
        df_label = equally_probable_intervals(price, classes)
    elif discretization == 'EOH': # 'every_one_hundred'
        df_label = every_one_hundred(price, classes)
    elif discretization == 'KM': # 'k_means'
        df_label = k_means(price, classes)
    elif discretization == 'KMI': # 'modify k_mean on interval'
        df_label = k_means_interval(price, classes, smallest_classes)
    elif discretization == 'KMN': # 'modify k_mean on number'
        df_label = k_means_number(price, classes, smallest_classes)
    elif discretization == 'KMD': # 'modify k_mean on density'
        df_label = k_means_density(price, classes, smallest_classes)
    else:
        print('no discretization')
    return df_label


# ### evaluate discretization by Coefficient of Variation


def find_variation_coefficient(df_label, classes):
    df_discretization = get_discretization_results(df_label, classes)
    number_mean = df_discretization['number'].mean()
    number_std = df_discretization['number'].std()
    number_cv = number_std / number_mean # cv = Coefficient of Variation
    interval_mean = df_discretization['interval'].mean()
    interval_std = df_discretization['interval'].std()
    interval_cv = interval_std / interval_mean # cv = Coefficient of Variation
    return number_cv, interval_cv


# ### campare all discretization by Coefficient of Variation


def find_number_interval(price, discretization, smallest_classes, biggest_classes):
    df_number_interval = pd.DataFrame(columns=[
        'number_mean', 'number_std', 'number_cv', 'interval_mean', 'interval_std', 'interval_cv'], 
                                      index=range(smallest_classes, biggest_classes))
    for classes in df_number_interval.index:
        df_label = choose_discretization(price, classes, discretization, smallest_classes)
        df_discretization = get_discretization_results(df_label, classes)
        number_mean = df_discretization['number'].mean()
        number_std = df_discretization['number'].std()
        number_cv = number_std / number_mean # cv = Coefficient of Variation
        interval_mean = df_discretization['interval'].mean()
        interval_std = df_discretization['interval'].std()
        interval_cv = interval_std / interval_mean # cv = Coefficient of Variation
        df_number_interval.loc[classes] = [
            number_mean, number_std, number_cv, interval_mean, interval_std, interval_cv]
    return df_number_interval


# #   classification



def get_cost_matrix(centroid,classes):
    cost_matrix = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            cost_matrix[i, j] = abs(centroid[i]-centroid[j])
    return cost_matrix



def calculate_cost(predictions, y_test, cost_matrix):
    cost_sum = 0
    for n in range(len(y_test)):
        cost_sum += cost_matrix[y_test[n], predictions[n]]
    cost = cost_sum/len(y_test)
    return cost



def prepare_data(data, df_label, classes):
    X_train, X_test, y_train, y_test = train_test_split(data, df_label, test_size=0.3, 
                                                        stratify=df_label['label'])
    # normalize based on train set, 'statify' is necessary
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    centroid = find_centroid(y_train, classes)
    cost_matrix = get_cost_matrix(centroid,classes)
    y_train = y_train['label'].values
    y_test = y_test['label'].values
    return X_train, X_test, y_train, y_test, cost_matrix



def train_test_once(X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, estimators):
    if cost_sensitive == False:
        rfc = RandomForestClassifier(n_estimators=estimators, n_jobs=-1)  # using all processors
        rfc.fit(X_train, y_train)
        predictions = rfc.predict(X_test)
        # use same function with cost-sensitive
    elif cost_sensitive == True:
        C_train = np.array([cost_matrix[i] for i in y_train])
        C_test = np.array([cost_matrix[i] for i in y_test])
        #### One-Vs-Rest, cost-weighting schema from WAP
        RFC_cs = WeightedOneVsRest(RandomForestClassifier(n_estimators=estimators, n_jobs=-1))
        RFC_cs.fit(X_train, C_train)
        predictions = RFC_cs.predict(X_test)
        # 'costsensitve' libarary has fit and predict method only
        # use same function with non cost-sensitive to calculate cost
    else:
        print('cost sensitive wrong')
    return predictions

def get_metrics(y_test, predictions, cost_matrix):
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions, average='macro')
    recall = metrics.recall_score(y_test, predictions, average='macro')
    f1_score = metrics.f1_score(y_test, predictions, average='macro')
    cost = calculate_cost(predictions, y_test, cost_matrix)
    return accuracy, precision, recall, f1_score, cost



def get_mean_results(X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, estimators, time):
    accuracy_list = []
    cost_list = []
    for i in range(time): # time = repeat time for experiments
        predictions = train_test_once(X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, 
                                      estimators)
        
        accuracy_ = np.mean(predictions==y_test)
        cost_ = calculate_cost(predictions, y_test, cost_matrix)
        accuracy_list.append(accuracy_)
        cost_list.append(cost_)
    accuracy = mean(accuracy_list)
    cost = mean(cost_list)
    return accuracy, cost


def run_multitime(data, price, smallest_classes, classes, discretization, cost_sensitive, estimators, time):
    df_label = choose_discretization(price, classes, discretization, smallest_classes)
    number, interval = find_variation_coefficient(df_label, classes)
    X_train, X_test, y_train, y_test, cost_matrix = prepare_data(data, df_label, classes)
    accuracy, cost = get_mean_results(X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, estimators, time)
    df_save = pd.DataFrame([[classes, discretization, cost_sensitive, number, interval, estimators, time, accuracy, cost]],
                           columns=['classes', 'discretization', 'cost_sensitive', 'number', 'interval', 'estimators', 'time', 'accuracy', 'cost'])
    df_save.to_csv(os.path.abspath('..') + '\\results\\results.csv',
                   mode='a', encoding='utf-8', index=False, header=False)
    print(classes, discretization, cost_sensitive, number, interval, estimators, time, accuracy, cost)

if __name__ == '__main__':
    data, price = input_row_data()
    estimators = 10
    time = 1
    smallest_classes = 4
    biggest_classes = 20
    classes_list = list(np.arange(smallest_classes, biggest_classes))
    discretization_list = ['EPI', 'EOH', 'KM', 'KMI', 'KMN', 'KMD']
    cost_sensitive = [False, True]
    for classes in classes_list:
        for discretization in discretization_list:
            df_label = choose_discretization(price, classes, discretization, smallest_classes)
            number, interval = find_variation_coefficient(df_label, classes)
            X_train, X_test, y_train, y_test, cost_matrix = prepare_data(data, df_label, classes)
            predictions = train_test_once(X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, estimators)
            accuracy, precision, recall, f1_score, cost = get_metrics(y_test, predictions, cost_matrix)
            df_save = pd.DataFrame([[classes, discretization, cost_sensitive, number, interval, estimators, time,
                                     accuracy, precision, recall, f1_score, cost]], columns=[
                'classes', 'discretization', 'cost_sensitive', 'number', 'interval', 'estimators', 'time',
                'accuracy', 'precision', 'recall', 'f1_score', 'cost'])
            df_save.to_csv(os.path.abspath('..') + '\\results\\results.csv',
                           mode='a', encoding='utf-8', index=False, header=False)
            print(classes, discretization, cost_sensitive, number, interval, estimators, time,
                  accuracy, precision, recall, f1_score, cost)
