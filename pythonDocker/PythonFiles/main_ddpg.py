#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:49:02 2017

@author: divyam
"""

import tensorflow as tf
import numpy as np
import tflearn
import yaml
import zmq
import time
import struct
import random
import math
import os.path
import multiprocessing
import argparse
import subprocess

from replaybuffer_ddpg import ReplayBuffer
from ExplorationNoise import ExplorationNoise
import leo_test

# ==========================
#   Training Parameters
# ==========================
# Max training steps
# MAX_EPISODES = 50000
# Max episode length
MAX_STEPS_EPISODE = 1010
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================

# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 64
MIN_BUFFER_SIZE = 1000
# Environment Parameters
ACTION_DIMS = 6
OBSERVATION_DIMS = 14
ACTION_BOUND = 2
# Noise Parameters
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2
RUNS = 1


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        actor_layer1 = tflearn.fully_connected(inputs, 400, activation='relu', name="actorLayer1")
        actor_layer2 = tflearn.fully_connected(actor_layer1, 300, activation='relu', name="actorLayer2")
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        actor_output = tflearn.fully_connected(actor_layer2, self.a_dim, activation='tanh', weights_init=w_init,
                                               name="actorOutput")
        scaled_output = tf.multiply(actor_output, self.action_bound)  # Scale output to -action_bound to action_bound
        return inputs, actor_output, scaled_output

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]
        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        critic_layer1 = tflearn.fully_connected(inputs, 400, activation='relu', name="criticLayer1")

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        critic_layer2 = tflearn.fully_connected(critic_layer1, 300, name="criticLayer2")
        critic_layer3 = tflearn.fully_connected(action, 300, name="criticLayerAction")

        net = tflearn.activation(tf.matmul(critic_layer1, critic_layer2.W) + tf.matmul(action, critic_layer3.W) +
                                 critic_layer3.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        critic_output = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, critic_output

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# ===========================
# Policy saving and loading
# ===========================


def open_config_file(conf):
    with open(conf, 'r') as f:
        config = yaml.load(f)
    return config


def check_for_policy_load(sess, config):
    if "load_file" in config["experiment"]:
        load_file = config["experiment"]["load_file"]
        if os.path.isfile(load_file):
            saver = tf.train.Saver()
            saver.restore(sess, load_file)
            print "Model Restored"
        else:
            print "Not a valid path"
            sess.run(tf.initialize_all_variables())
    else:
        sess.run(tf.initialize_all_variables())

    return sess


def check_for_policy_save(config):
    save_every = config["experiment"]["save_every"]

    if save_every == "never":
        save_counter = 0
    elif save_every == "trail":
        save_counter = 1
    elif save_every == "test":
        save_counter = config["experiment"]["test_interval"] + 1
    else:
        save_counter = 50
    return save_counter


def compute_action(test_agent, actor, mod_state, noise):
    if test_agent:
        action = actor.predict(np.reshape(mod_state, (1, actor.s_dim)))
        time.sleep(0.05)
    else:
        action = actor.predict(np.reshape(mod_state, (1, actor.s_dim))) + noise

    action = np.reshape(action, (ACTION_DIMS,))

    action = np.clip(action, -ACTION_BOUND, ACTION_BOUND)
    clip_action = action
    return action, clip_action


def get_address(config):
    address = config['experiment']['agent']['communicator']['addr']
    address = address.split(':')[-1]
    address = "tcp://*:{}".format(address)

    return address


# ===========================
#   Agent Training
# ===========================
def train(args, sess, actor, critic):
    # Parse the configuration file
    config = open_config_file(args)

    # Check if a policy needs to be loaded
    sess = check_for_policy_load(sess, config)

    # Check if a policy needs to be saved
    save_counter = check_for_policy_save(config)
    print save_counter

    # writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Initialize constants for exploration noise
    episode_count = 0
    ou_sigma = OU_SIGMA

    # Establish the connection
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.bind(get_address(config))

    while True:

        ep_reward = 0
        ep_ave_max_q = 0
        mod_old_state = np.zeros(actor.s_dim)
        computed_action = np.zeros(ACTION_DIMS)
        terminal = 0
        noise = np.zeros(actor.a_dim)

        while True:

            # Receive the state from zeromq, the current state
            incoming_message = server.recv()

            # Get the length of the message
            len_incoming_message = len(incoming_message)

            # Decide which method sent the message and extract the message in a numpy array
            if len_incoming_message == (OBSERVATION_DIMS + 1) * 8:
                a = np.asarray(struct.unpack('d' * (OBSERVATION_DIMS + 1), incoming_message))
                test_agent = a[0]
                state = a[1: OBSERVATION_DIMS + 1]
                # mod_state = np.array([math.cos(state[0]), math.sin(state[0]), state[1]])
                mod_state = state
                episode_start = True
                episode_count += 1

            else:
                a = np.asarray(struct.unpack('d' * (OBSERVATION_DIMS + 3), incoming_message))
                test_agent = a[0]
                state = a[1: OBSERVATION_DIMS + 1]
                # mod_state = np.array([math.cos(state[0]), math.sin(state[0]), state[1]])
                mod_state = state
                reward = a[OBSERVATION_DIMS + 1]
                terminal = a[OBSERVATION_DIMS + 2]
                episode_start = False

            # Add the transition to replay buffer
            if not episode_start and not test_agent:
                if terminal == 2:
                    replay_buffer.add(np.reshape(mod_old_state, (actor.s_dim,)),
                                      np.reshape(computed_action, (actor.a_dim,)),
                                      reward, True, np.reshape(mod_state, (actor.s_dim,)))
                else:
                    replay_buffer.add(np.reshape(mod_old_state, (actor.s_dim,)),
                                      np.reshape(computed_action, (actor.a_dim,)),
                                      reward, False, np.reshape(mod_state, (actor.s_dim,)))

            # Compute OU noise
            noise = ExplorationNoise.ou_noise(OU_THETA, OU_MU, ou_sigma, noise, ACTION_DIMS)
            scaled_noise = noise * ACTION_BOUND

            # Compute action
            computed_action, clip_action = compute_action(test_agent, actor, mod_state, scaled_noise)

            # Convert action into null terminated string 
            action_message = struct.pack('d' * ACTION_DIMS, *clip_action)

            # Sends the predicted action via zeromq
            server.send(action_message)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MIN_BUFFER_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            # old_state = state

            mod_old_state = mod_state

            if not episode_start:
                ep_reward += reward

            if episode_start and episode_count != 1:
                print "Episode Ended:", ep_reward, episode_count, "-", multiprocessing.current_process().name
                if save_counter != 0 and episode_count % save_counter == 0 and episode_count != 0:
                    saver = tf.train.Saver()
                    saver.save(sess, "model-mp-{}".format(multiprocessing.current_process().name))
                    ou_sigma *= 0.99
                break


def main(args):
    with tf.Session() as sess:
        # Initialize the actor and critic networks
        actor = ActorNetwork(sess, OBSERVATION_DIMS, ACTION_DIMS, ACTION_BOUND, \
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, OBSERVATION_DIMS, ACTION_DIMS, CRITIC_LEARNING_RATE, TAU,
                               actor.get_num_trainable_vars())

        # Train the network
        train(args, sess, actor, critic)

if __name__ == '__main__':
    # tf.app.run()
    # parse arguments

    pool = multiprocessing.Pool(1)
    pool.map(main, leo_test.rl_run_zmqagent(['PythonFiles/leo_zmqagent.yaml'], range(RUNS)))
