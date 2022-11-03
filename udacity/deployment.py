# from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


def read_paths():
    ##################Load config.json and correct path variable
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    model_path = os.path.join(config['output_model_path'])

    return dataset_csv_path, prod_deployment_path, model_path


####################function for deployment
def store_model_into_pickle(model_path, dataset_csv_path, prod_deployment_path):
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    trained_model = pickle.load(open(model_path + 'trainedmodel.pkl', 'rb'))
    pickle.dump(trained_model, open(prod_deployment_path + 'trainedmodel.pkl', 'wb'))

    with open(dataset_csv_path + 'latestscore.txt') as file:
        model_score = file.readlines()

    myfile = open(prod_deployment_path + 'latestscore.txt', 'w')
    for element in model_score:
        myfile.write('%s \n' % str(element))

    with open(dataset_csv_path + 'ingestedfiles.txt') as file:
        ingested_data = file.readlines()

    myfile = open(prod_deployment_path + 'ingestedfiles.txt', 'w')
    for element in ingested_data:
        myfile.write('%s' % str(element))


def deploy():
    dataset_csv_path, prod_deployment_path, model_path = read_paths()

    store_model_into_pickle(model_path=model_path, dataset_csv_path=dataset_csv_path,
                            prod_deployment_path=prod_deployment_path)


if __name__ == '__main__':
    dataset_csv_path, prod_deployment_path, model_path = read_paths()

    store_model_into_pickle(model_path=model_path, dataset_csv_path=dataset_csv_path,
                            prod_deployment_path=prod_deployment_path)
