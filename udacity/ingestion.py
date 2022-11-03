import pandas as pd
import numpy as np
import os
import json
import subprocess
from datetime import datetime


def read_paths():
    #############Load config.json and get input and output paths
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    return input_folder_path, output_folder_path


#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # check for datasets, compile them together, and write to an output file
    final_dataframe = pd.DataFrame(
        columns=['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'])
    filenames = os.listdir(input_folder_path)
    for each_filename in filenames:
        currentdf = pd.read_csv(input_folder_path + each_filename)
        final_dataframe = final_dataframe.append(currentdf).reset_index(drop=True)

    final_dataframe.drop_duplicates(inplace=True)

    assert final_dataframe.duplicated().sum() == 0
    filename = 'finaldata.csv'
    final_dataframe.to_csv(output_folder_path + filename)

    outputlocation = output_folder_path + 'ingestedfiles.txt'

    myfile = open(outputlocation, 'w')
    for element in filenames:
        myfile.write('%s \n' % str(element))

    return


def perform_ingestion():
    input_folder_path, output_folder_path = read_paths()

    merge_multiple_dataframe(input_folder_path, output_folder_path)
    return


if __name__ == '__main__':
    input_folder_path, output_folder_path = read_paths()

    merge_multiple_dataframe(input_folder_path, output_folder_path)
