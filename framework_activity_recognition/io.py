import yaml
import os
from pathlib import Path
import logging

def load_config_file(file_path):
    """
    Read and load yaml file based on the given path
    Argument:
        file_path: path to the yaml file
    """
    with open(file_path, "r") as ymlfile:
        #config = yaml.safe_load(ymlfile)
        config = yaml.load(ymlfile, Loader=yaml.Loader)
    return config

def make_model_dir(config_file):
    """
    Create directory to the experiment folder based on given path in config file
    Argument:
        config_file: configuration file
    """
    experiment_config = config_file["experiment"]
    model_dir = Path(experiment_config["model_save_path"]) / experiment_config["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def make_logging_dir(config_file):
    """
    Create folder to store log and pth files based on given path in config file
    Argument:
        config_file: configuration file
    """
    experiment_config = config_file["experiment"]
    logging_dir = Path(experiment_config["model_save_path"]) / experiment_config["name"] / ("exp" + str(config_file["experiment"]["experiment_number"])) 
    logging_dir.mkdir(parents=True, exist_ok=True)
    return logging_dir

def make_benchmark_dir(config_file):
    """
    Create folder to testing log files based on given path in config file
    Argument:
        config_file: configuration file
    """
    architecture_config = config_file["architecture"]
    benchmark_dir = Path(architecture_config["model"]) / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir


def make_logger(logger_path):
    """
    Create logger file based on given path in config file
    Argument:
        config_file: configuration file
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s :: %(message)s")

    if logger_path is not None:
        file_handler = logging.FileHandler(logger_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logging.getLogger("").addHandler(stream_handler)
    return logger

