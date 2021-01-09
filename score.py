import json
import numpy as np
import os
import pickle
import joblib
import pandas
import traceback
from azureml.core.model import Model



def init():
    global model
    # Get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)



def run(raw_data):
    try:
        #data = np.array(json.loads(raw_data)) 
        data = json.loads(raw_data)['data']
        input_data = pandas.DataFrame.from_dict(data)
        result = model.predict(input_data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return traceback.format_exc()




