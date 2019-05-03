"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


n_zeros = 1 # should be same to train n_zeros


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:\\Users\\DamonChang\\Desktop\\KETI_pose\\edm2d\\experiments\\base_model')
parser.add_argument('--data_dir', default='D:\\Human pose DB\\data\\img')
parser.add_argument('--restore_from', default='best_weights')

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, 'zeros_{}'.format(n_zeros), 'tt')

    # Get the filenames from the test set

    test_filenames = [os.path.join(test_data_dir, 'img', f) for f in os.listdir(test_data_dir + '\\' + 'img')[:200]]
    test_labels = [os.path.join(test_data_dir, 'label', f) for f in os.listdir(test_data_dir + '\\' + 'label')[:200]]


    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
