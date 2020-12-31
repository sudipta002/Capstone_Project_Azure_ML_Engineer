import json
import numpy as np
import os
import joblib
import traceback


def init():
    global model
    path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    model = joblib.load(path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return traceback.format_exc()