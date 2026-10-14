import os
import yaml
import torch
import logging
import subprocess
from collections import deque
from logging.handlers import RotatingFileHandler

from models.AdapGL import AdapGLA


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_configs():
    configs_folder = 'configs'
    return {
        "model": load_config(os.path.join(configs_folder, 'model_config.yaml')),
        "train": load_config(os.path.join(configs_folder, 'train_config.yaml')),
        "app": load_config(os.path.join(configs_folder, 'app_config.yaml')),
        "data": load_config(os.path.join(configs_folder, 'data_config.yaml'))
    }


def prepare_directories(configs):
    os.makedirs(configs["data"]['DATA_FOLDER'], exist_ok=True)
    os.makedirs(configs["model"]['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(configs["data"]['FEATURE_HISTORY_FOLDER'], exist_ok=True)
    os.makedirs(configs["data"]['LOGS_FOLDER'], exist_ok=True)


def setup_logger(module_name, log_size, backup_count):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    log_file = f"{module_name}.log" if module_name != "__main__" else "main.log"
    handler = RotatingFileHandler(log_file, maxBytes=log_size, backupCount=backup_count)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def create_model(model_config, num_nodes, device):
    return AdapGLA(
        num_block=model_config['NUM_BLOCKS'],
        num_nodes=num_nodes,
        step_num_in=model_config['STEP_NUM_IN'],
        step_num_out=model_config['STEP_NUM_OUT'],
        input_size=model_config['INPUT_SIZE'],
        num_cheb_filter=8,
        num_time_filter=8,
        conv_type='gcn',
        K=2
    ).to(device)


def initialize_buffers(service_names, history_length, inference_track_len=10):
    node_buffers = {name: deque(maxlen=history_length) for name in service_names}
    inference_times = deque(maxlen=inference_track_len)
    service_name_to_node_id = {name: idx for idx, name in enumerate(service_names)}
    return node_buffers, inference_times, service_name_to_node_id


def get_device():
    return torch.device('cpu')


def extract_constants(configs):
    return {
        "NUM_NODES": configs["app"]['NUM_NODES'],
        "INPUT_SIZE": configs["model"]['INPUT_SIZE'],
        "STEP_NUM_IN": configs["model"]['STEP_NUM_IN'],
        "STEP_NUM_OUT": configs["model"]['STEP_NUM_OUT'],
        "NUM_BLOCKS": configs["model"]['NUM_BLOCKS'],
        "ADJ_TYPE": configs["model"]['ADJ_TYPE'],
        "DATA_FOLDER": configs["data"]['DATA_FOLDER'],
        "SLEEP_INTERVAL": configs["app"]['SLEEP_INTERVAL'],
        "MODEL_PATH": configs["model"]['MODEL_PATH'],
        "MODEL_FOLDER": configs["model"]['MODEL_FOLDER'],
        "MAX_LOG_SIZE": configs["model"]['MAX_LOG_SIZE'],
        "BACKUP_COUNT": configs["model"]['BACKUP_COUNT'],
        "FEATURE_HISTORY_FOLDER": configs["data"]['FEATURE_HISTORY_FOLDER'],
        "HISTORY_LENGTH": configs["data"]['HISTORY_LENGTH'],
        "BATCH_SIZE": configs["train"]['BATCH_SIZE'],
        "NUM_EPOCHS": configs["train"]['NUM_EPOCHS'],
        "LEARNING_RATE": configs["train"]['LEARNING_RATE'],
        "NUM_THREADS": configs["train"]['NUM_THREADS'],
        "NUM_DATALOADER_WORKERS": configs["train"]['NUM_DATALOADER_WORKERS'],
        "FACTOR": configs["train"]['FACTOR'],
        "PATIENCE": configs["train"]['PATIENCE'],
        "NUM_MODELS": configs["model"]['NUM_MODELS']
    }
