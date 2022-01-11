"""
Main module for running model training.
"""

import argparse
import logging
import sys
import time
from collections import namedtuple
from datetime import datetime
from functools import partial

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, FeatureUnion

from lib.scripts.global_variables import (MODEL_NAME, TRAIN_DATA_PATH, TEST_DATA_PATH,
                                          CHECKPOINTS_PATH, TRAIN_DATA_CH, TEST_DATA_CH,
                                          FEATURE_COL, PREPR_FEATURE_COL, LABEL_COL,
                                          CATEGORIES_COL, LOG_PATH, N_CORES,
                                          vectorizer_params, linear_model_params)
from lib.scripts.preprocessing import Preprocessor
from lib.scripts.reader import save_model, load_model
from lib.scripts.utils import seed_everything, get_report

seed_everything(10)


class Train:
    def __init__(self, logger_settings: namedtuple, use_df_checkpoints: bool, df_parallel: bool):
        self.model_name = MODEL_NAME
        self.trained_model_name: str = None
        self.train_data = TRAIN_DATA_PATH
        self.test_data = TEST_DATA_PATH
        self.checkpoints = CHECKPOINTS_PATH
        self.feature = FEATURE_COL
        self.categories_col = CATEGORIES_COL
        self.prepr_feature = PREPR_FEATURE_COL
        self.label = LABEL_COL

        self.use_df_checkpoints = use_df_checkpoints
        self.train_data_ch = TRAIN_DATA_CH
        self.test_data_ch = TEST_DATA_CH

        self.preprocessor = Preprocessor()
        self.df_parallel = df_parallel
        self.n_cores = N_CORES

        self.report = get_report
        self.logger = self._get_logger(logger_settings, log_path=LOG_PATH)

    def _get_logger(self, logger_settings: namedtuple, log_path: str) -> logging.RootLogger:
        """Returns logger with pre-defined settings."""
        logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        if logger_settings.debug:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            logger.addHandler(stdout_handler)

        if logger_settings.log_to_file:
            file_handler = logging.FileHandler(f"{log_path}/{datetime.now().strftime('%Y_%m_%d')}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(logging.DEBUG)

        return logger

    def run(self) -> None:
        """Runs model training."""

        if not self.use_df_checkpoints:
            self.logger.debug('Loading train and test data...')
            df_train = pd.read_csv(self.train_data, usecols=[self.feature, self.label])
            df_test = pd.read_csv(self.test_data, usecols=[self.feature, self.categories_col, self.label])

            self.logger.debug('Extracting symbols for processing...')
            symbols_to_move = self.preprocessor.extract_symbols(df_train,
                                                                feature_col_name=self.feature)

            # with open(f"{self.checkpoints}symbols_to_move.pickle", 'wb') as f:
            #     pickle.dump(symbols_to_move, f)
            save_model(symbols_to_move, f"{self.checkpoints}symbols_to_move.pickle")

            # with open(f"{self.checkpoints}/symbols_to_move.pickle", 'rb') as f:
            #     symbols_to_move = pickle.load(f)
            symbols_to_move = load_model(f"{self.checkpoints}/symbols_to_move.pickle")

            self.logger.debug('Train data processing...')
            if self.df_parallel:
                df_train = self.preprocessor.df_parallel_prepr_run(df=df_train,
                                                                   feature_colname=self.feature,
                                                                   prepr_feature_colname=self.prepr_feature,
                                                                   f=partial(self.preprocessor.preproc,
                                                                             sym=symbols_to_move),
                                                                   n_cores=self.n_cores)
            else:
                df_train[self.prepr_feature] = \
                    df_train[self.feature].apply(self.preprocessor.preproc, sym=symbols_to_move)

            self.logger.debug('Test data processing...')
            df_test[self.prepr_feature] = \
                df_test[self.feature].apply(self.preprocessor.preproc, sym=symbols_to_move)

            self.logger.debug('Saving preprocessed dataframes to csv files...')
            df_train.to_csv(f"{self.checkpoints}{self.train_data_ch}", index=False, encoding='utf8')
            df_test.to_csv(f"{self.checkpoints}{self.test_data_ch}", index=False, encoding='utf8')

        else:
            self.logger.debug('Loading train and test data checkpoints...')
            df_train = pd.read_csv(f"{self.checkpoints}{self.train_data_ch}", encoding='utf8', keep_default_na=False)
            df_test = pd.read_csv(f"{self.checkpoints}{self.test_data_ch}", encoding='utf8', keep_default_na=False)

        self.logger.debug('Vectorization text data and setting model parameters')

        vec = FeatureUnion([("tfidf-1", TfidfVectorizer(**vectorizer_params['tfidf_unigram'])),
                            ("tfidf-2", TfidfVectorizer(**vectorizer_params['tfidf_bigram']))])

        clf = LogisticRegression(**linear_model_params, max_iter=200)
        pipe = make_pipeline(vec, clf)

        self.logger.debug('Model fitting...')
        pipe.fit(df_train[self.prepr_feature].values.tolist(),
                 df_train[self.label].values)

        self.trained_model_name = \
            f"{self.checkpoints}{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}.pickle"

        self.logger.debug(f'Model saving')
        # with open(self.trained_model_name, 'wb') as f:
        #     pickle.dump(pipe, f)
        save_model(pipe, self.trained_model_name)

        # with open(self.trained_model_name, 'rb') as f:
        #     pipe = pickle.load(f)
        pipe = load_model(self.trained_model_name)

        report = self.report(pipe, df_test, categories_col_name=self.categories_col)

        self.logger.debug("Model metrics are following:\n")
        self.logger.debug("".join(f"\n{metric}" for metric in report))
        self.logger.debug(f"The model has been trained and saved to {self.trained_model_name}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=True, help='Enable logging to stdout.')
    parser.add_argument('-f', '--log_to_file', type=bool, default=True, help='Enable logging to file.')
    parser.add_argument('-c', '--use_df_c', type=bool, default=False, help='Enable using dataframes checkpoints.')
    parser.add_argument('-p', '--df_parallel', type=bool, default=True, help='Enable parallel df processing.')
    args = parser.parse_args()

    Logger_settings = namedtuple('Logger_settings', ('debug', 'log_to_file'))

    train = Train(Logger_settings(args.debug, args.log_to_file),
                  use_df_checkpoints=args.use_df_c,
                  df_parallel=args.df_parallel)
    train.run()
