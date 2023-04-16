import yaml
import joblib
from datetime import datetime
import os

# Construct the path to the config file using the util.py file path
util_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.abspath(os.path.join(util_dir, '..', 'config', 'config.yaml'))

def time_stamp() -> datetime:
    return datetime.now()

def load_config() -> dict:
    try:
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError(f"Parameters file not found in {config_dir}")
    
    return config

def pickle_load(file_path: str):
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    joblib.dump(data, file_path)

params = load_config()
PRINT_DEBUG = params['print_debug']

def print_debug(messages: str) -> None:
    if PRINT_DEBUG == True:
        print(time_stamp(), messages)