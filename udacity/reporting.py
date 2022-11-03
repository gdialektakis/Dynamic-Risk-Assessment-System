import pickle
import subprocess

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from diagnostics import model_predictions
import json
import os

###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    test_data_path = os.path.join(config['test_data_path'])
    test_data = pd.read_csv(test_data_path + 'testdata.csv')

    prod_deployment_path = os.path.join(config['prod_deployment_path'])

    # trained_model = pickle.load(open(prod_deployment_path + 'trainedmodel.pkl', 'rb'))

    y_pred, y_test = model_predictions(test_data, prod_deployment_path)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_plot = sns.heatmap(conf_matrix, annot=True)
    fig = conf_matrix_plot.get_figure()
    output_path = os.path.join(config['output_model_path'])
    fig.savefig(output_path + "confusionmatrix.png")
    # plt.show()
    return


def run():
    score_model()


if __name__ == '__main__':
    score_model()
