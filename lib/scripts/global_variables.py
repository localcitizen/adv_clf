# coding:utf8
"""
Module with global variables for the project.
"""

from pathlib import Path

from lib.scripts.reader import read_yml

TRAIN_MODEL_CONFIG = Path(__file__).resolve().parent.parent / 'configs/train_model_config.yml'

train_model_config = read_yml(TRAIN_MODEL_CONFIG)
train_model_config = train_model_config['main']

MODEL_NAME = train_model_config['model_name']

# path settings
train_model_config_paths = train_model_config['paths']
TRAIN_DATA_PATH = train_model_config_paths['train_data']
TEST_DATA_PATH = train_model_config_paths['test_data']
CHECKPOINTS_PATH = train_model_config_paths['checkpoints']

# logging path
LOG_PATH = train_model_config_paths['log_path']

# preprocessing checkpoints path
text_preproc_paths = train_model_config_paths['text_preproc_files']
STOP_WORDS_PATH = text_preproc_paths['stop_words']
WORD_TO_DIGIT_PATH = text_preproc_paths['word_to_digit']

# checkpoints path
TRAIN_DATA_CH = train_model_config_paths['train_data_ch']
TEST_DATA_CH = train_model_config_paths['test_data_ch']

# dataset settings
DATA_COLUMNS = train_model_config['data_columns']
POSTFIX = '_prepr'
FEATURE_COL = DATA_COLUMNS['feature']
PREPR_FEATURE_COL = FEATURE_COL + POSTFIX
CATEGORIES_COL = DATA_COLUMNS['categories']
LABEL_COL = DATA_COLUMNS['label']

# model settings
model_params = train_model_config['model_params']
vectorizer_params = model_params['TfidfVectorizer']
linear_model_params = model_params['LogisticRegression']

# the amount of cores for parallel computing
N_CORES = train_model_config['n_cores']

# unit test config
UNIT_TEST_CONFIG = Path(__file__).resolve().parent.parent / 'configs/test_model_config.yml'

unit_test_config = read_yml(UNIT_TEST_CONFIG)
unit_test_config = unit_test_config['main']['test']

unit_test_config_path = unit_test_config['paths']
unit_test_config_checkpoints = unit_test_config['checkpoints']

UNIT_TEST_MODEL_PATH = unit_test_config_path['model']
UNIT_TEST_SCENARIOS_PATH = unit_test_config_path['scenarios']

UNIT_TEST_MODEL_CHECKPOINT = unit_test_config_checkpoints['model']
UNIT_TEST_SCENARIOS_CHECKPOINT = unit_test_config_checkpoints['scenarios']
