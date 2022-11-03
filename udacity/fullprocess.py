
import ast
import json
import os
import pickle

import pandas as pd
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls

################## Check and read new data
# first, read ingestedfiles.txt
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
model_path = os.path.join(config['output_model_path'])
ingested_data_path = os.path.join(config['output_folder_path'])

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_data = os.listdir(input_folder_path)

################## Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
proceed = False
with open(prod_deployment_path + 'ingestedfiles.txt') as ingestedFile:
    for file in source_data:
        if str(file) not in ingestedFile.read():
            print('File not exists')
            proceed = True
        else:
            print('File already exists. Exiting!')
            exit()

model_drift = False
if proceed:
    ingestion.perform_ingestion()

    ################## Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    previous_score = []
    f = open(prod_deployment_path + 'latestscore.txt')
    for line in f.readlines():
        previous_score.append(float(line))

    # TODO: This needs to change. Has to be the new ingested data from prevous step.
    test_data = pd.read_csv(ingested_data_path + 'finaldata.csv')

    trained_model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))
    f1_score = scoring.score_model(trained_model, ingested_data_path + 'finaldata.csv', ingested_data_path)

    print('Previous f1 score: {}\n'.format(previous_score[0]))

    if f1_score < previous_score[0]:
        print('\nModel drift has occurred!\n')
        model_drift = True


################## Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if model_drift:
    # Retrain the model
    print('\n----------- Training on New Data ----------\n')
    training.perform_training()

    ################## Re-deployment
    # if you found evidence for model drift, re-run the deployment.py script
    print('\n----------- Deployment ----------\n')
    deployment.deploy()

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    print('\n----------- Diagnostics ----------\n')
    diagnostics.run_diagnostics()
    print('\n----------- Reporting ----------\n')
    reporting.run()
    print('\n----------- API Calls ----------\n')
    apicalls.run_api_calls()









