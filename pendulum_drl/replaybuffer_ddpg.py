#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""
from collections import deque
import random
import numpy as np
import yaml
import os
import sys
import pickle
import math


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.transitions_count = 0
        self.buffer_count = 0
        self.buffer = deque()
        self.transitions = deque()
        self.save = 0
        self.load = 0
        self.diff = 0
        self.save_filename = None
        self.load_filename = None
        self.model_filename = None
        random.seed(random_seed)
        self.read_cfg('config.yaml')

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.buffer_count < self.buffer_size:
            self.buffer.append(experience)
            self.buffer_count += 1

        else:
            self.buffer.popleft()
            self.buffer.append(experience)

        if self.buffer_count == self.buffer_size:
            if self.save:
                with open(self.save_filename, 'w') as f:
                    pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        return False

    def size(self):
        return self.buffer_count

    def sample_batch(self, batch_size):
        batch = []

        if self.buffer_count < batch_size:
            batch = random.sample(self.buffer, self.buffer_count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.transitions_count = 0
        self.buffer_count = 0

    def read_cfg(self, cfg):
        yfile = '/home/divyam/rl-agent/pendulum_drl/%s' % cfg
        if os.path.isfile(yfile) == False:
            print 'File %s not found' % yfile
        else:
            # open configuration
            stream = file(yfile, 'r')
            conf = yaml.load(stream)
            self.save = int(conf['replay_buffer']['save'])
            self.load = int(conf['replay_buffer']['load'])
            self.diff = int(conf['difference_model'])
            print self.save, self.load, self.diff
            if self.save == 1:
                self.save_filename = conf['replay_buffer']['save_filename']
            if self.load == 1:
                self.load_filename = conf['replay_buffer']['load_filename']
                with open(self.load_filename) as f:
                    self.transitions = pickle.load(f)
                    f.close()
            if self.diff == 1:
                self.model_filename = conf['difference_model']['model_filename']
            stream.close()

    def sample_state_action(self, state, action, check):

        if self.load == 0:
            return state, action

        if self.load == 1:
            temp = self.transitions[self.transitions_count]

            state = temp[0]
            action = temp[1]
            self.transitions_count += 1
            return state, action


