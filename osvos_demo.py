"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""

import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
slim = tf.contrib.slim

# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)

# constants
KNOWN_SETS = ['bear', 'blackswan', 'bmx-bumps', 'boat', 'breakdance', 'breakdance-flare', 'bus', 'dog']


def run_demo(seq_name, max_training_iters=500,  **kwargs):
    # User Defined parameters
    gpu_id = kwargs.get('gpu_id', 0)
    train_model = kwargs.get('train_model', True)
    side_supervision = kwargs.get('side_supervision', 1)

    # define paths
    result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
    parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
    logs_path = os.path.join('models', seq_name)

    # Train parameters
    learning_rate = kwargs.get('learning_rate', 1e-8)
    save_step = kwargs.get('save_step', max_training_iters)
    display_step = kwargs.get('display_step', 10)





    # Define Dataset
    test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
    test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
    if train_model:
        train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                      os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
        dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
    else:
        dataset = Dataset(None, test_imgs, './')

    # Train the network
    if train_model:
        # More training parameters

        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(gpu_id)):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                     save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

    # Test the network
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
            osvos.test(dataset, checkpoint_path, result_path)


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser('parameters for running OSVOS')
    parser.add_argument('seq_name', choices=KNOWN_SETS, help='the name of the sequence to run')
    # parse arguments
    args = parser.parse_args()
    seq = args.seq_name
    kwargs = vars(args)
    del kwargs['seq_name']
    print('Parameters:', kwargs)
    # run learner
    run_demo(seq, **kwargs)
