import argparse
import os
import logging

from model.utils import Params
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.training import train_and_evaluate
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='D:\\Human pose DB\\data\\user')
parser.add_argument('--model_dir', default='C:\\Users\\DamonChang\\Desktop\\KETI_pose\\edm2d\\experiments\\base_model')
parser.add_argument('--data_dir', default='D:\\Human pose DB\\data\\img')
parser.add_argument('--restore_from', default=None)
# parser.add_argument('--restore_from', default='C:\\Users\\DamonChang\\Desktop\\KETI_pose\\edm2d\\experiments\\base_model\\last_weights')

# train_subjects = [1, 5, 6, 7, 8]  # [1,5,6,7,8]
# test_subjects = [9, 11] # [9,11]
n_zeros = 1  # 1, 2, 3
set_num_data = -1 # 400, 600, -1(all)



if __name__ == '__main__':
    # the number of occlusion, randomly make zero values in EDM

    ## json params load
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    set_logger(os.path.join(args.model_dir, 'train.log'))


    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir,'zeros_{}'.format(n_zeros),'tr')
    dev_data_dir = os.path.join(data_dir,'zeros_{}'.format(n_zeros),'dev')

    train_filenames = [os.path.join(train_data_dir, 'img', f) for f in os.listdir(train_data_dir+'\\'+'img')[:set_num_data]]
    eval_filenames = [os.path.join(dev_data_dir, 'img', f) for f in os.listdir(dev_data_dir+'\\'+'img')[:set_num_data]]

    train_labels = [os.path.join(train_data_dir, 'label', f) for f in os.listdir(train_data_dir+'\\'+'label')[:set_num_data]]
    eval_labels = [os.path.join(dev_data_dir, 'label', f) for f in os.listdir(dev_data_dir+'\\'+'label')[:set_num_data]]

    assert len(train_labels) == len(train_filenames)  and len(eval_filenames) == len(eval_labels)

    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)
    params.num_zeros = n_zeros

    # apply function to each sample
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

    # model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    logging.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)































