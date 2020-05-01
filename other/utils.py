from __future__ import absolute_import

import argparse
import logging

from sklearn.metrics import f1_score, recall_score, precision_score


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=
                                """
                                Command line arguments
                                """,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--mode", type=str, choices=['train', 'test'], help="Mode of the model")
    p.add_argument("--model_path", type=str, help="model_path")
    p.add_argument("--init_checkpoint", default='/disk1/sajad/pretrained-bert/pretraining_output', type=str,
                   help="model_path")
    p.add_argument("--log_file", type=str, default='log.txt', help="model_path")
    p.add_argument("--data_dir", type=str, help="model_path")
    p.add_argument("--saved_model", type=str, help="saved_model_path")
    p.add_argument("--max_len", type=int, default=256, help="Max length of input sequence")
    p.add_argument("--epochs", type=int, default=10, help="Epochs")
    p.add_argument("--seed", type=int, default=88, help="Initial seed")
    p.add_argument("--batch_size", type=int, default=5, help="Batch size")
    p.add_argument("--device_id", type=int, default=0, help="GPU device id")
    p.add_argument("--report_every", type=int, default=50, help="Report interval (steps)")
    p.add_argument("--task", type=str, default='a', help="Sub-task")
    return p.parse_args()


logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def evaluate_model_normal(y_test, y_pred):
    score = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    return score, precision, recall
