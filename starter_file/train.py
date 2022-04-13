import os
import joblib
import argparse
import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from azureml.core.dataset import Dataset
from azureml.data.datapath import DataPath

ds = TabularDatasetFactory.from_delimited_files("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")

run = Run.get_context()

def clean_data(data):
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop("name", inplace=True, axis=1)
    y_df = x_df.pop("status")
    return x_df, y_df

 
x, y = clean_data(ds)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.28, random_state=42)
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")
    
    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()