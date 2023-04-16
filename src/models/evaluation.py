# Get the grandparent directory of the current script (../../.py)
import sys
import os
from pathlib import Path
grandparent_dir = str(Path(__file__).resolve().parents[2]) # change to parent 2 (two) levels
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
os.chdir(grandparent_dir)

import src.util as utils
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_data(config):
    X_train_feng = utils.pickle_load(config['train_feng_set_path'][0])
    y_train_feng = utils.pickle_load(config['train_feng_set_path'][1])
    X_test_feng = utils.pickle_load(config['test_feng_set_path'][0])
    y_test_feng = utils.pickle_load(config['test_feng_set_path'][1])
    return X_train_feng, y_train_feng, X_test_feng, y_test_feng

def load_model(config, type='knn'):
    if type == 'knn':
        clf = utils.pickle_load(config['knn_model_path'])
    elif type == 'rf': 
        clf = utils.pickle_load(config['production_model_path'])
    else:
        raise Exception("invalid model type!")
    
    return clf

def check_cross_val_score(clf, X_train, y_train, verbose=False):
    n_splits=5
    cv = StratifiedKFold(n_splits=n_splits)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    if verbose:
        print(f"k-Fold ({n_splits} fold(s)) CV Score for Random Forest:")
        print("Cross-validation scores:", cv_scores)
        print("Mean cross-validation accuracy:", np.mean(cv_scores))

    return cv_scores, np.mean(cv_scores)

def check_test_score(config, clf, X_test, y_test, verbose=False):
    y_pred = clf.predict(X_test)
    test_acc_score = clf.score(X_test, y_test)
    report = classification_report(y_test, y_pred, target_names=config['encoder_classes'])
    if verbose:
        print("Accuracy score of the model (on test set):", test_acc_score)
        print(report)

    return test_acc_score, report

def check_roc_auc(config, clf):
    y_score = clf.predict_proba(X_test_feng)

    n_classes = len(config['encoder_classes'])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_feng == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc

if __name__ == '__main__':
    print("### MODEL EVALUATION START")
    print("1. Config loading", end=' - ')
    config = utils.load_config()
    print("Done")
    
    print("2. Data loading", end=' - ')
    X_train_feng, y_train_feng, X_test_feng, y_test_feng = load_data(config)
    print("Done")

    print("3a. RandomForest Model loading", end=' - ')
    rf_clf = load_model(config, 'rf')
    print("Done")

    print("3b. kNN Model loading", end=' - ')
    knn_clf = load_model(config, 'knn')
    print("Done")

    print("4a. Checking Cross-Val score for RandomForest")
    rf_cv_scores, rf_mean_cv_scores = check_cross_val_score(rf_clf, X_train_feng, y_train_feng, verbose=True)
    print("4b. Checking Cross-Val score for kNN")
    knn_cv_scores, knn_mean_cv_scores = check_cross_val_score(knn_clf, X_train_feng, y_train_feng, verbose=True)
    print("5a. Checking Test Set score for RandomForest")
    rf_test_scores, rf_classification_report = check_test_score(config, rf_clf, X_test_feng, y_test_feng, verbose=True)
    print("5b. Checking Test Set score for kNN")
    knn_test_scores, knn_classification_report = check_test_score(config, knn_clf, X_test_feng, y_test_feng, verbose=True)
    
    print("6a. ROC AUC for RandomForest")
    rf_roc_auc = check_roc_auc(config, rf_clf)
    print(rf_roc_auc)

    print("6b. ROC AUC for kNN")
    knn_roc_auc = check_roc_auc(config, knn_clf)
    print(knn_roc_auc)
