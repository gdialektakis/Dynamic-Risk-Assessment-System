import os
import requests


def run_api_calls():
    # Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1:8000"

    TEST_DATA_PATH = 'testdata/'
    MODEL_PATH = 'models/'

    response1 = requests.post(f'{URL}/prediction', json={'dataset_path': os.path.join(TEST_DATA_PATH, 'testdata.csv')})
    print(response1.status_code)
    response2 = requests.get(f'{URL}/scoring')
    print(response2.status_code)
    response3 = requests.get(f'{URL}/summarystats')
    print(response3.status_code)
    response4 = requests.get(f'{URL}/diagnostics')
    print(response4.status_code)

    with open(os.path.join(MODEL_PATH, 'apireturns.txt'), 'w') as file:
        file.write('Model Predictions\n')
        file.write(response1.text)
        file.write('\nModel Score\n')
        file.write(response2.text)
        file.write('\nSummary Statistics\n')
        file.write(response3.text)
        file.write('\nDiagnostics\n')
        file.write(response4.text)

    return