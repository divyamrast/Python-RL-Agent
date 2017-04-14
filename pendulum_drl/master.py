#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
from main_ddpg import start
import yaml
import tensorflow as tf
from train_model import training
from train_model import prediction
import pickle
import numpy as np


def main():
    # Run the training on the ideal model phase
    cfg = "/home/divyam/grl/cfg/ZeromqAgent/pendulum_drl_1.yaml"
    a = start(cfg)
    time.sleep(5)

    # for i in range(5):
    #     # Run the trained policy on a real model
    #     d = {'replay_buffer': {'load': 0, 'save': 1, 'save_filename': 'saved_data-1'},
    #          'difference_model': 0}
    #     with open('config.yaml', 'w') as yaml_file:
    #         yaml.dump(d, yaml_file, default_flow_style=False)
    #     cfg_mod = "/home/divyam/grl/cfg/ZeromqAgent/pendulum_drl_2.yaml"
    #     start(cfg_mod)
    #     time.sleep(2)
    #
    #     # Run the transitions on the original model
    #     d = {'replay_buffer': {'load': 1, 'load_filename': 'saved_data-1', 'save': 1, 'save_filename': 'saved_data-2'},
    #          'difference_model': 0}
    #     with open('config.yaml', 'w') as yaml_file:
    #         yaml.dump(d, yaml_file, default_flow_style=False)
    #     cfg = "/home/divyam/grl/cfg/ZeromqAgent/pendulum_drl_3.yaml"
    #     start(cfg)
    #
    #     # # Train a neural network or update one
    #     sess = tf.Session()
    #     d = {'replay_buffer': {'load': 0, 'save': 0},'difference_model': 0}
    #     with open('config.yaml', 'w') as yaml_file:
    #         yaml.dump(d, yaml_file, default_flow_style=False)
    #     model = training(sess, 'saved_data-1', 'saved_data-2', 4, 3)
    #
    #     # Prediction (temporary)
    #     with open('saved_data-2') as f:
    #         transitions = pickle.load(f)
    #         f.close()
    #     s = np.array([_[0] for _ in transitions])
    #     a = np.array([_[1] for _ in transitions])
    #     s2 = np.array([_[4] for _ in transitions])
    #     predicted_obs, predicted_state = prediction(model, np.concatenate((s, a, s2), axis=1), 4, 3)
    #     b = np.concatenate((s, a, predicted_state), axis=1)
    #     np.savetxt('saved_data-3.csv',b,delimiter=',',newline='\n')
    #
    #     # Checking prediction with mse
    #     with open('saved_data-1') as f:
    #         transitions = pickle.load(f)
    #         f.close()
    #     s2 = np.array([_[4] for _ in transitions])
    #     print(((predicted_state[8001:10000,:] - s2[8001:10000,:]) ** 2).mean(axis = 0))

    # sess.close()

if __name__ == '__main__':
    main()
