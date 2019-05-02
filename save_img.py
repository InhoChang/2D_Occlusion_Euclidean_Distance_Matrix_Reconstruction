import argparse
import h5py
import os
import tensorflow as tf
import numpy as np
import copy
import random

from collections import OrderedDict
from model.utils import Params
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='D:\\Human pose DB\\data\\user')
parser.add_argument('--model_dir', default='C:\\Users\\DamonChang\\Desktop\\KETI_pose\\edm2d\\experiments\\base_model')
parser.add_argument('--save_path', default='D:\\Human pose DB\\data\\img')

train_subjects = [1, 5, 6, 7, 8]  # [1,5,6,7,8]
test_subjects = [9, 11]  # [9,11]


def load_data(root_path, subjects):

    data = OrderedDict()

    for subj in subjects:
        path2d = os.path.join(root_path, 'S{}'.format(subj), 'annot2d.h5')

        with h5py.File(path2d, 'r+') as h5f:
            edm2d = h5f['edm_2d'][:]

        for i in range(len(edm2d)):
            data[subj, i] = edm2d[i]

    return data


def save_img(input, save_path, n_zeros, idx):
    # save_img(data, output_dir_split, n_zeros, i)

    ## change value  between 0 ~ 255
    # also change dtype as from float64 to uint8
    input = np.reshape(input, [16, 16])
    input_255 = (input * 255).astype(np.uint8)
    label_255 = copy.deepcopy(input_255)

    assert sum(sum(input_255 - label_255)) == 0

    ## make zeros
    for i in random.sample(range(len(input_255)), n_zeros):

        for j in range(len(input_255)):
            input_255[i][j] = 0
            input_255[j][i] = 0

    Image.fromarray(input_255).save(save_path + '\\' + 'img' + '\\' + 'zeros_{}_img_{}.png'.format(n_zeros, idx))
    Image.fromarray(label_255).save(save_path + '\\' + 'label' + '\\' + 'zeros_{}_label_{}.png'.format(n_zeros, idx))


if __name__ == '__main__':

    # the number of occlusion, randomly make zero values in EDM
    n_zeros = 2  # 1, 2, 3

    ## json params load
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    ## data load
    train_data = load_data(args.root_path, train_subjects)
    test_data = load_data(args.root_path, test_subjects)

    tf.set_random_seed(230)
    tr2d = copy.deepcopy(np.vstack(train_data.values()))

    rand_idx = list(range(len(tr2d)))
    random.shuffle(rand_idx)
    split = int(0.8 * len(rand_idx))

    tr_inputs = tr2d[rand_idx[:split]]
    dev_inputs = tr2d[rand_idx[split:]]
    test_inputs = copy.deepcopy(np.vstack(test_data.values()))

    params.train_size = len(tr_inputs)
    params.eval_size = len(dev_inputs)
    params.test_size = len(test_inputs)

    params.num_zeros = n_zeros

    inputs = {'tr': tr_inputs,
              'dev': dev_inputs,
              'tt': test_inputs}

    ## save_img process
    for split in ['tr', 'dev', 'tt']:

        output_dir_split = os.path.join(args.save_path, 'zeros_{}'.format(n_zeros), '{}'.format(split))
        i = 0

        for data in tqdm(inputs[split], desc=split):
            save_img(data, output_dir_split, n_zeros, i)
            i += 1




# ## check img
# img_a = Image.open('D:\\Human pose DB\\data\\img\\zeros_1\\tr\\img\\zeros_1_img_10.png')
# label_a = Image.open('D:\\Human pose DB\\data\\img\\zeros_1\\tr\\label\\zeros_1_label_10.png')
# img_array = np.array(img_a)
# label_array = np.array(label_a)















