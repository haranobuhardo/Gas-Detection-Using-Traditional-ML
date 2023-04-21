# Get the grandparent directory of the current script (../../.py)
import sys
import os
from pathlib import Path
grandparent_dir = str(Path(__file__).resolve().parents[2]) # change to parent 2 (two) levels
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
os.chdir(grandparent_dir)

import src.util as utils
import yaml
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(config):
    X_train = utils.pickle_load(config['train_set_path'][0])
    y_train = utils.pickle_load(config['train_set_path'][1])
    X_test = utils.pickle_load(config['test_set_path'][0])
    y_test = utils.pickle_load(config['test_set_path'][1])
    X_val = utils.pickle_load(config['val_set_path'][0])
    y_val = utils.pickle_load(config['val_set_path'][1])

    return X_train, X_val, X_test, y_train, y_val, y_test

def outlier_removal(column: pd.Series):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return column[(column > lower_bound) & (column < upper_bound)]

def label_encoder(y, target_classes, save_encoder_classes=False, config_file=None):
    encoder = LabelEncoder()
    encoder.fit(target_classes)
    y_label_encoded = encoder.transform(y)

    if save_encoder_classes and config_file:
        # config_dir = os.path.abspath(os.path.join(util_dir, '..', 'config', 'test.yaml'))
        with open(os.path.join('config', config_file), 'r') as file:
            config = yaml.safe_load(file)
            config.update({'encoder_classes': encoder.classes_.tolist()})
        with open(os.path.join('config', config_file), 'w') as file:
            documents = yaml.safe_dump(config, file)

    return y_label_encoded

def standard_scaler(X, scaler = None):
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(X)
    
    X_scaled = scaler.transform(X)

    return X_scaled, scaler


if __name__ == '__main__':
    print("### DATA PREPROCESSING START")
    print("1. Config loading", end=' - ')
    config = utils.load_config()
    print("Done")


    print("2. Data loading (X, y)", end=' - ')
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config)
    print("Done")

    print(X_test)
    print(type(X_test))

    
    print("3. Outliers Removal", end=' - ')
    for column in config['predictors']:
        X_train[column] = outlier_removal(X_train[column])
    y_train = y_train[X_train.notna().all(axis=1)]
    X_train = X_train.dropna(how='any')
    print("Done")

    print("4. Standard Scaling", end=' - ')
    X_train_feng, scaler = standard_scaler(X_train)
    utils.pickle_dump(scaler, config['scaler_path']) # save scaler after using training data
    X_test_feng, _ = standard_scaler(X_test, scaler)
    X_val_feng, _ = standard_scaler(X_val, scaler)
    print("Done")


    print("5. Label Encoding", end=' - ')
    y_test_feng = label_encoder(y_test.values.ravel(), config['target_classes'], save_encoder_classes=True, config_file='config.yaml')
    y_train_feng = label_encoder(y_train.values.ravel(), config['target_classes'])
    y_val_feng = label_encoder(y_val.values.ravel(), config['target_classes'])
    print("Done")

    print("6. Pickle dumping (X, y feng version)", end=' - ')
    utils.pickle_dump(X_train_feng, config['train_feng_set_path'][0])
    utils.pickle_dump(y_train_feng, config['train_feng_set_path'][1])
    utils.pickle_dump(X_test_feng, config['test_feng_set_path'][0])
    utils.pickle_dump(y_test_feng, config['test_feng_set_path'][1])
    utils.pickle_dump(X_val_feng, config['val_feng_set_path'][0])
    utils.pickle_dump(y_val_feng, config['val_feng_set_path'][1])
    print("Done")
