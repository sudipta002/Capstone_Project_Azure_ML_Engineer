import os
import joblib
import argparse
import numpy as np
import pandas as pd

from azureml.core.run import Run
from azureml.core import Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_data(input_data, ws):
    # Get the input dataset by name
    dataset = Dataset.get_by_name(ws, name=input_data)

    # Load the TabularDataset to pandas DataFrame
    df = dataset.to_pandas_dataframe()

    X = df.drop(columns=['Class']).copy(deep=True)
    y = df['Class'].copy(deep=True)

    #Dataset split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

    return x_train, x_test, y_train, y_test


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization.")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge.")
    parser.add_argument("--input_data", type=str, help="Name of dataset in ML Studio.")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    ws = run.experiment.workspace
    x_train, x_test, y_train, y_test = load_data(args.input_data, ws)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()