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
    #################Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    model_path = os.path.join(config['output_model_path'])
    return dataset_csv_path, test_data_path, model_path


#################Function for model scoring
def score_model(trained_model, test_data_path, dataset_csv_path):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    test_data = pd.read_csv(test_data_path)
    if len(test_data.axes[1]) == 6:
        # Drop index
        test_data = test_data.iloc[:, 1:]
    test_data.drop('corporation', axis=1, inplace=True)
    y_test = test_data.pop('exited')
    X_test = test_data

    y_pred = trained_model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)
    print('F1 score: {}'.format(f1_score))

    output_file = 'latestscore.txt'
    myfile = open(dataset_csv_path + output_file, 'w')
    myfile.write('%s \n' % str(f1_score))
    return f1_score


def run_scoring():
    dataset_csv_path, test_data_path, model_path = read_paths()
    trained_model = pickle.load(open(model_path + 'trainedmodel.pkl', 'rb'))
    f1_score = score_model(trained_model=trained_model, test_data_path=test_data_path+'testdata.csv',
                           dataset_csv_path=dataset_csv_path)

    return f1_score


if __name__ == '__main__':
    dataset_csv_path, test_data_path, model_path = read_paths()
    trained_model = pickle.load(open(model_path + 'trainedmodel.pkl', 'rb'))
    f1_score = score_model(trained_model=trained_model, test_data_path=test_data_path+'testdata.csv',
                           dataset_csv_path=dataset_csv_path)
