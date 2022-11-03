import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


##################Load config.json and get environment variables
def read_paths():
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    return dataset_csv_path, test_data_path, prod_deployment_path


##################Function to get model predictions
def model_predictions(test_data, prod_deployment_path=None):
    prod_deployment_path = 'production_deployment/'
    trained_model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))
    # trained_model = pickle.load(open(prod_deployment_path + 'trainedmodel.pkl', 'rb'))
    # read the deployed model and a test dataset, calculate predictions
    test_data.drop('corporation', axis=1, inplace=True)
    y_test = test_data.pop('exited')
    X_test = test_data

    y_pred = trained_model.predict(X_test)
    return y_pred, y_test


##################Function to get summary statistics
def dataframe_summary():
    with open('config.json', 'r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'])
    df = pd.read_csv(dataset_csv_path + 'finaldata.csv')
    df = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    # calculate summary statistics here
    themeans = list(df.mean())
    themedians = list(df.median())
    stdevs = list(df.std())

    summary = []
    summary.append(themeans)
    summary.append(themedians)
    summary.append(stdevs)
    return summary


##################Function to get timings
def execution_time():
    final_output = []
    # calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing = timeit.default_timer() - starttime
    print('Ingestion timing: {}'.format(ingestion_timing))

    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - starttime
    print('Training timing: {}'.format(training_timing))

    result_list = [
        {'ingest_time': ingestion_timing},
        {'train_time': training_timing}
    ]
    return result_list


def check_missing_data():
    with open('config.json', 'r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'])
    df = pd.read_csv(dataset_csv_path + 'finaldata.csv')
    df = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    nas = list(df.isna().sum())
    nap_ercents = [nas[i] / len(df.index) for i in range(len(nas))]
    return nap_ercents


def outdated_packages_list():

    dependencies = subprocess.run(['pip', 'list', '--outdated'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

    dep = dependencies.stdout
    dep = dep.translate(str.maketrans('', '', ' \t\r'))
    dep = dep.split('\n')
    dep = [dep[3]] + dep[5:-3]

    return dep


def run_diagnostics():
    dataset_csv_path, test_data_path, prod_deployment_path = read_paths()
    test_data = pd.read_csv(test_data_path + 'testdata.csv')
    model_predictions(test_data, prod_deployment_path)
    dataframe_summary()
    execution_time()
    outdated_packages_list()


if __name__ == '__main__':
    dataset_csv_path, test_data_path, prod_deployment_path = read_paths()
    test_data = pd.read_csv(test_data_path + 'testdata.csv')
    model_predictions(test_data, prod_deployment_path)
    dataframe_summary()
    execution_time()
    outdated_packages_list()
