# Get the grandparent directory of the current script (../../.py)
import sys
import os
from pathlib import Path
grandparent_dir = str(Path(__file__).resolve().parents[2]) # change to parent 2 (two) levels
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
os.chdir(grandparent_dir)

import src.util as utils

import json
from datetime import datetime as dt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def load_data(config):
    X_train_feng = utils.pickle_load(config['train_feng_set_path'][0])
    y_train_feng = utils.pickle_load(config['train_feng_set_path'][1])
    X_val_feng = utils.pickle_load(config['val_feng_set_path'][0])
    y_val_feng = utils.pickle_load(config['val_feng_set_path'][1])

    return X_train_feng, X_val_feng, y_train_feng, y_val_feng

def time_stamp(to_str = False):
    if to_str:
        return dt.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return dt.now()

def log_json(current_log: dict, log_path: str):
    current_log = current_log.copy()

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log

def log_json(logs: dict, file_path: str):
    try:
        # Check if the file exists, and create it if it doesn't
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass

        # Log the JSON data
        with open(file_path, 'a') as f:
            json.dump(logs, f)
            f.write('\n')  # Add a newline for better readability of the log file

    except Exception as e:
        print(f"Error while logging JSON data: {e}")

def call_grid_search(estimator, param_grid, X, y):
    # Grid search for RandomForest hyperparameters

    print("Doing GridSearch based on:", param_grid)    
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    return grid_search

def save_grid_search_log(grid_search_cv_result, config):
    _grid_search_result_df = pd.DataFrame(grid_search_cv_result)
    _grid_search_result_df['timestamp'] = time_stamp(to_str=True)
    log_json(_grid_search_result_df.to_dict(), config['hyperparameter_tuning_log_path'])

def call_modeling(type:str, params, X_train, y_train, X_val, y_val):
    if type == 'knn':
        clf = KNeighborsClassifier(**params)
    elif type == 'rf':
        clf = RandomForestClassifier(**params, random_state=42)
    else:
        raise Exception("Invalid model type!")
    
    clf.fit(X_train, y_train)

    if len(X_val) > 0 and len(y_val) > 0:
        best_val_score = clf.score(X_val, y_val)
        print(f'Best model accuracy score on val set: {best_val_score}')

    return clf
    

if __name__ == '__main__':
    print("### DATA MODELING START")
    print("1. Config loading", end=' - ')
    config = utils.load_config()
    print("Done")

    print("2. Data loading (X,y val and train)", end=' - ')
    X_train_feng, X_val_feng, y_train_feng, y_val_feng = load_data(config)
    print("Done")

    print("3a. Grid Search on RandomForest")
    _ = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search_rf = call_grid_search(_, param_grid_rf, X_train_feng, y_train_feng)
    save_grid_search_log(grid_search_rf.cv_results_, config)
    print("GridSearch log saved!")
    
    print("3b. Grid Search on kNN")
    _ = KNeighborsClassifier()
    param_grid_knn = {
        'n_neighbors': [5, 10, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [10, 30, 50]
    }
    grid_search_knn = call_grid_search(_, param_grid_knn, X_train_feng, y_train_feng)
    save_grid_search_log(grid_search_rf.cv_results_, config)
    print("GridSearch log saved!")

    print("4a. Modeling for RandomForest")
    rf_clf = call_modeling('rf', grid_search_rf.best_params_, X_train_feng, y_train_feng, X_val_feng, y_val_feng)

    print("4b. Modeling for kNN")
    knn_clf = call_modeling('knn', grid_search_knn.best_params_, X_train_feng, y_train_feng, X_val_feng, y_val_feng)

    print("5. Model pickle dumping", end=' - ')
    utils.pickle_dump(rf_clf, config['production_model_path'])
    utils.pickle_dump(knn_clf, config['knn_model_path'])
    print("Done")