# Get the grandparent directory of the current script (../../.py)
import sys
import os
from pathlib import Path
grandparent_dir = str(Path(__file__).resolve().parents[2]) # change to parent 2 (two) levels
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
os.chdir(grandparent_dir)

## boilerplate
from src import util as utils
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(config):
    df = pd.read_csv(config['dataset_path'])
    df = df.loc[:, config['predictors'] + [config['label']]]
    return df

def predictors_type_conversion(input_data, config):
    for column in config['int_columns']:
        input_data[column] = input_data[column].astype('int32')

    return input_data

def check_data(input_data, config):
    # Measure the length of the data
    len_input_data = len(input_data)

    # Check data types
    assert input_data.select_dtypes("int").columns.to_list() == config['int_columns'], "an error occurred in int columns"
    assert input_data.select_dtypes("object").columns.to_list() == config['obj_columns'], "an error occurred in object columns"

    # Check target classes
    assert input_data[config['label']].value_counts().index.to_list() == config['target_classes'], "an error occurred in target classes check"

    # Check sensor values range
    for i in input_data.filter(regex='MQ.*').columns.to_list():
        assert input_data[i].between(config['range_sensor_val'][0], config['range_sensor_val'][1]).sum() == len_input_data, "an error occurred in sensor values range check"

def data_splitting(df, config):
    # Split data into train and test sets
    X = df.drop(config['obj_columns'], axis=1)
    y = df[config['obj_columns']]
    X_traival, X_test, y_traival, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_traival, y_traival, test_size=0.25, random_state=42, stratify=y_traival)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    print("### DATA PREPARATION START")
    print("1. Load config", end=' - ')
    config = utils.load_config()
    print("Done")
    print("2. Load dataset", end=' - ')
    df = load_data(config)
    print("Done")
    print("3. Data type conversion", end=' - ')
    df = predictors_type_conversion(df, config)
    print("Done")
    print("3. Data defense (check and test dataframe)", end=' - ')
    check_data(df, config)
    print("Done")
    print("4. Data splitting", end=' - ')
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(df, config)
    print("Done")

    print("5. Pickle Dumping Dataset, X_train, y_train, X_test, y_test, X_val, y_val", end=' - ')
    utils.pickle_dump(df, config['dataset_processed_path'])
    utils.pickle_dump(X_train, config['train_set_path'][0])
    utils.pickle_dump(y_train, config['train_set_path'][1])
    utils.pickle_dump(X_test, config['test_set_path'][0])
    utils.pickle_dump(y_test, config['test_set_path'][1])
    utils.pickle_dump(X_val, config['val_set_path'][0])
    utils.pickle_dump(y_val, config['val_set_path'][1])
    print("Done")