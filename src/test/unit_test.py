# Get the grandparent directory of the current script (../../.py)
import sys
import os
from pathlib import Path
grandparent_dir = str(Path(__file__).resolve().parents[2]) # change to parent 2 (two) levels
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
os.chdir(grandparent_dir)

import src.util as utils
import src.data.preparation as preparation
import src.features.preprocessing as preprocessing
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# Load config
config = utils.load_config()

def test_type_conversion():
    test_data = {
        "MQ2": [278],
        "MQ3": [295],
        "MQ5": [302],
        "MQ6": ['675'],
        "MQ7": [237],
        "MQ8": ['566'],
        "MQ135": [761],
    }

    test_df = pd.DataFrame(test_data)
    expec_df = pd.DataFrame({
        "MQ2": [278], 
        "MQ3": [295], 
        "MQ5": [302], 
        "MQ6": [675],
        "MQ7": [237], 
        "MQ8": [566],
        "MQ135": [761]
        }, dtype="int32")
    
    result_df = preparation.predictors_type_conversion(test_df, config)
    assert expec_df.equals(result_df)

def test_label_encoder():
    test_y = np.array(['Mixture', 'Mixture', 'Perfume', 'Smoke', 'Perfume', 'NoGas'])
    expc_y = np.array([0, 0, 2, 3, 2, 1])
    result_y = preprocessing.label_encoder(test_y, config['target_classes'])
    print(result_y, expc_y)

    assert np.array_equal(result_y, expc_y)

def test_standard_scaler():
    test_X = [[278, 295, 302, 675, 237, 566, 761]]
    scaler = utils.pickle_load(config['scaler_path'])
    scaled_X, _ = preprocessing.standard_scaler(pd.DataFrame(test_X), scaler)
    expc_X = np.array([[-4.32343032, -2.35037379, -1.85922757, 6.31966859, -3.99977757, 0.1739888, 4.57282051]])
    # print(scaled_X)
    assert np.allclose(scaled_X, expc_X, atol=1e-8)


if __name__ == '__main__':
    test_type_conversion()
    test_label_encoder()
    test_standard_scaler()