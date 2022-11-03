from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


def read_paths():
    ###################Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    return dataset_csv_path, model_path


#################Function for training the model
def train_model(data, model_path):
    # use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)

    # fit the logistic regression to your data
    data.drop('corporation', axis=1, inplace=True)
    data = data.iloc[:, 1:]
    y = data.pop('exited')
    X = data
    lr.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    path = model_path + 'trainedmodel.pkl'
    pickle.dump(lr, open(path, 'wb'))
    return


def perform_training():
    dataset_csv_path, model_path = read_paths()
    data_df = pd.read_csv(dataset_csv_path + 'finaldata.csv')
    train_model(data=data_df, model_path=model_path)


if __name__ == '__main__':
    dataset_csv_path, model_path = read_paths()
    data_df = pd.read_csv(dataset_csv_path + 'finaldata.csv')
    train_model(data=data_df, model_path=model_path)
