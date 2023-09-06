import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

def generate_test_data(a: float, b: float, num_samples: int, mean: float, std_dev: float):
    x = np.random.uniform(0, 1, num_samples)
    errors = np.random.normal(mean, std_dev, num_samples)
    y = a*x + b + errors
    return x, y


def preprocess_data(data: pd.DataFrame):
    # Shuffle the data rows randomly and extract feature and target values.
    data = data.sample(frac=1)
    # Extract features and targets
    features = data.iloc[:,:-1].to_numpy()
    targets = data['Strength'].to_numpy()
    # Standardize feature values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
    
    return (X_train, X_test, y_train, y_test)

def data_generator(data:pd.DataFrame, batch_size:int):
    # Process the raw data and extract the features and targets
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    # Batch the data
    correct_divs = [i for i in range(1,len(X_train)) if len(X_train)%i==0]
    # Raise error if batch_size is not appropiate.
    if len(X_train)%batch_size != 0:
        raise ValueError(f"Inappropiate batch size! Dataset could not be evenly partitioned.\nSize of dataset: {len(X_train)}\nPossible batch sizes: {correct_divs}")
    # Partition list of items into batches
    batched_X_train = np.array_split(X_train, len(X_train)//batch_size)
    batched_y_train = np.array_split(y_train, len(y_train)//batch_size)
    
    training_data = ([torch.tensor(batched_X_train[i]), torch.tensor(batched_y_train[i])] for i in range(len(batched_X_train)))
    testing_data = [X_test, y_test]
    
    return (training_data, testing_data)
    
    
    


