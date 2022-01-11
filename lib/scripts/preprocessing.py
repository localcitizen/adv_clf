"""
Module with text preprocessing functions.
"""

import multiprocessing as mp
import os
import re
import signal
import time
from itertools import chain as ch
from typing import Callable

import numpy as np
import pandas as pd

from lib.scripts.global_variables import STOP_WORDS_PATH, WORD_TO_DIGIT_PATH
from lib.scripts.reader import read_json

stop_words = read_json(STOP_WORDS_PATH)
word_to_digit_mapping = read_json(WORD_TO_DIGIT_PATH)

stop_words = list(ch(stop_words['single_character'],
                     stop_words['double_character'],
                     stop_words['double_character_words'],
                     stop_words['words']))


class Preprocessor:
    """Class with text preprocessing methods."""

    @staticmethod
    def extract_symbols(df: pd.DataFrame, feature_col_name: str) -> list:
        """Extracts a set of symbols from train dataset.

        Args:
             df: Dataframe containing text column be predicted.
             feature_col_name: Feature column name.

        Returns:
            List with the extracted symbols.
        """
        sym = [''.join(re.sub('\d+|\w+', '', s).split()) for s in df[feature_col_name]]
        sym = set(''.join(sym))
        sym.difference_update({'@'})
        sym = list(sym)

        return sym

    @staticmethod
    def word_to_digit_transformation(line: str) -> str:
        """Transforms text representation of digits to symbol.

        Args:
            line: Input text line.

        Returns:
            Text with digits in symbols notation (given the digits in text line).
        """
        for s in word_to_digit_mapping['hundreds']:
            line = line.replace(s, word_to_digit_mapping['hundreds'][s])

        for s in word_to_digit_mapping['tens']:
            line = line.replace(s, word_to_digit_mapping['tens'][s])

        for s in word_to_digit_mapping['singles']:
            line = line.replace(s, word_to_digit_mapping['singles'][s])

        return line

    @staticmethod
    def preproc(sentence: str, sym: list) -> str:
        """Preprocesses input text.

        Args:
            sentence: Input text.
            sym: List of symbols to be removed in the input text.

        Returns:
            Preprocessed text.
        """
        line = re.sub(r'\n|/|\\|%|!|\(|\)', ' ', sentence).strip()

        line = line.lower()
        line.replace('точка', '.')

        line = ' '.join(re.split(r'(-?\d*\.?\d+)', line))

        for s in sym:
            line = line.replace(s, ' ')

        line = re.sub(' +', ' ', line)

        line = line.split(' ')
        line = [word for word in line if word not in stop_words]

        line = ' '.join(line)

        line = Preprocessor.word_to_digit_transformation(line)

        line = re.sub('8', 'восемь', line)
        line = re.sub('9', 'девять', line)
        line = re.sub('\d', 'цифра', line)

        return line

    @staticmethod
    def df_parallel_prepr_run(df: pd.DataFrame,
                              feature_colname: str,
                              prepr_feature_colname: str,
                              f: Callable,
                              n_cores: int):
        """Runs dataframe parallel processing.

        Args:
            df: Input dataframe to be preprocessed.
            feature_colname: Dataframe column containing feature.
            prepr_feature_colname: Dataframe column containing preprocessed feature.
            f: Function to be applied to dataframe feature column.
            n_cores: Amount of cores for parallel processing.

        Returns:
            Preprocessed dataframe.
        """
        df_chunks = np.array_split(df, n_cores)
        queue, procs, keywords_df = [], [], {}

        for i, chunk in enumerate(df_chunks):
            queue.append(mp.Queue())
            keywords_df[i] = {"df_chunk": df_chunks[i], "feature_colname": feature_colname,
                              "prepr_feature_colname": prepr_feature_colname,
                              "queue": queue[i], "func": f, "num": i}

            p = mp.Process(target=Preprocessor._df_apply, kwargs=keywords_df[i])
            procs.append(p)
            p.start()

        res = {}
        done_parts = 0
        while done_parts <= i:
            time.sleep(5)
            for q in queue:
                if q.empty():
                    continue
                else:
                    done_parts += 1
                    res.update(q.get())
                    time.sleep(1)

        # TODO: Delete
        time.sleep(5)

        for proc in procs:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGTERM)

        del keywords_df

        df = pd.concat(res[i] for i in range(len(df_chunks)))
        return df

    @staticmethod
    def _df_apply(df_chunk: pd.DataFrame, feature_colname: str, prepr_feature_colname: str,
                  queue: mp.Queue, func: Callable, num: int) -> None:
        df_chunk[prepr_feature_colname] = df_chunk[feature_colname].apply(lambda t: func(t))
        queue.put({num: df_chunk})
