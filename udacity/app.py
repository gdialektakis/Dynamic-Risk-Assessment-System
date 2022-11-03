from flask import Flask, jsonify, request
import pandas as pd
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, check_missing_data
from scoring import run_scoring
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3

    filepath = request.get_json()['dataset_path']

    df = pd.read_csv(filepath)
    predictions, y_true = model_predictions(df)
    return jsonify(predictions.tolist())


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    # check the score of the deployed model
    f1 = run_scoring()
    return str(f1)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    summary = dataframe_summary()
    return json.dumps(summary)


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    # check timing and percent NA values
    exec_time = execution_time()
    outdated = outdated_packages_list()
    na_info = check_missing_data()
    result = {
        'missing_percentage': na_info,
        'execution_time': exec_time,
        'outdated_packages': outdated
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
