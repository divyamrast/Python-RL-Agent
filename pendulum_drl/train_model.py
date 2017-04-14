#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn
import pickle
import math
import random
import yaml
import os


class DifferenceModel(object):
    def __init__(self, sess, input_dim, output_dim):
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = 0.01
        self.inputs, self.outputs = self.create_network()
        self.actual_output = tf.placeholder(tf.float32, [None, self.output_dim])

        self.loss = tflearn.mean_square(self.actual_output, self.outputs)

        self.optimize = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def create_network(self):
        inputs = tflearn.input_data(shape=[None, self.input_dim])
        layer1 = tflearn.fully_connected(inputs, 400, activation='relu', name="Layer1",
                                         weights_init=tflearn.initializations.uniform(
                                             minval=-1 / math.sqrt(self.input_dim),
                                             maxval=1 / math.sqrt(self.input_dim)),
                                         regularizer='L2')
        layer2 = tflearn.fully_connected(layer1, 300, activation='relu', name="Layer2",
                                         weights_init=tflearn.initializations.uniform(minval=-1 / math.sqrt(400),
                                                                                      maxval=1 / math.sqrt(400)),
                                         regularizer='L2')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(layer2, self.output_dim, activation='linear', weights_init=w_init,
                                         name="Output")
        return inputs, output

    def train(self, inputs, actual_output):
        self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: inputs,
            self.actual_output: actual_output
        })

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs
        })


def training_data(file1, file2):
    randomize = []

    with open(file1) as f:
        transitions = pickle.load(f)
        f.close()
        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        s2 = np.array([_[4] for _ in transitions])
        b1 = np.concatenate((s, a, s2), axis=1)

    with open(file2) as f:
        transitions = pickle.load(f)
        f.close()

        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        s2 = np.array([_[4] for _ in transitions])
        b2 = np.concatenate((s, a, s2), axis=1)
        return b1, b2


def training(sess, file1, file2, input_dim, output_dim):
    # Initialize neural net
    model = DifferenceModel(sess, input_dim, output_dim)

    # Load data
    data_1, data_2 = training_data(file1, file2)

    state = data_1[:, 0:input_dim - 1]
    action = np.reshape(data_1[:, input_dim - 1], (data_1.shape[0], 1))
    next_state_1 = data_1[:, input_dim :input_dim + output_dim ]
    next_state_2 = data_2[:, input_dim :input_dim + output_dim ]

    obs = invert_state_obs(state)
    next_obs_1 = invert_state_obs(next_state_1)
    next_obs_2 = invert_state_obs(next_state_2)

    diff_obs = next_obs_1 - next_obs_2
    diff_state = next_state_1 - next_state_2
    print (((next_state_1[8001:10000,:] - next_state_2[8001:10000,:]) ** 2).mean(axis = 0))
    data = np.concatenate((state[1:8000,:], action[1:8000,:], diff_state[1:8000,:]), axis=1)
    np.random.shuffle(data)
    # sample = random.sample(data, 100)
    sess.run(tf.initialize_all_variables())
    print(data.shape)

    # Train the model
    for i in range(800):
        model.train(data[10*i:10*(i+1),0:input_dim], data[10*i:10*(i+1),input_dim:input_dim+output_dim])

    # Save the model
    saver = tf.train.Saver()
    saver.save(sess, "model-diff-{}".format(1))

    return model


def prediction(model, inputs, input_dim, output_dim):
    # Predict using the model
    state = inputs[:, 0:input_dim - 1]
    action = np.reshape(inputs[:, input_dim - 1], (inputs.shape[0], 1))
    next_state = inputs[:, input_dim:input_dim + output_dim]
    obs = invert_state_obs(state)
    next_obs = invert_state_obs(next_state)
    state_output = model.predict(np.concatenate((state, action), axis=1)) + next_state
    obs_output = invert_state_obs(state_output)

    return obs_output, state_output


def invert_state_obs(a):
    b = np.zeros((np.shape(a)[0],2))
    for i in range(np.shape(a)[0]):
        b[i][:] = np.array([math.atan2(a[i][1], a[i][0]), a[i][2]])
    return b


def invert_obs_state(a):
    b = np.zeros((np.shape(a)[0],3))
    for i in range(np.shape(a)[0]):
        b[i][:] = np.array([math.cos(a[i][0]), math.sin(a[i][0]), a[i][1]])
    return b
