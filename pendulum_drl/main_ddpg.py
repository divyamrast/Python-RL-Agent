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
import sys
from subprocess import Popen
import signal

from replaybuffer_ddpg import ReplayBuffer
from ExplorationNoise import ExplorationNoise
from actor import ActorNetwork
from critic import CriticNetwork

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
TRAINING_SIZE = 10000
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64
MIN_BUFFER_SIZE = 1000
# Environment Parameters
ACTION_DIMS = 1
OBSERVATION_DIMS = 2
ACTION_BOUND = 1
ACTION_BOUND_REAL = 3
# Noise Parameters
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


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
        meta_file = "{}.meta".format(load_file)
        if os.path.isfile(meta_file):
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
        save_counter = 10
    return save_counter


def compute_action(test_agent, actor, mod_state, noise):
    if test_agent:
        action = actor.predict(np.reshape(mod_state, (1, actor.s_dim)))
        # time.sleep(0.05)
    else:
        action = actor.predict(np.reshape(mod_state, (1, actor.s_dim))) + noise

    action = np.reshape(action, (ACTION_DIMS,))

    action = np.clip(action, -1, 1)

    return action


def get_address(config):
    address = config['experiment']['agent']['communicator']['addr']
    address = address.split(':')[-1]
    address = "tcp://*:{}".format(address)

    return address


def invert(state):
    return np.array([math.atan2(state[1], state[0]),state[2]])


# ===========================
#   Agent Training
# ===========================

def train(args, sess, actor, critic):

    # Start the GRL code
    code = Popen(['../../grl/build/grld', args])

    # Parse the configuration file
    config = open_config_file(args)

    # Check if a policy needs to be loaded
    sess = check_for_policy_load(sess, config)

    # Check if a policy needs to be saved
    save_counter = check_for_policy_save(config)
    print save_counter

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    replay_buffer_test = ReplayBuffer(TRAINING_SIZE, RANDOM_SEED)

    # Initialize constants for exploration noise
    episode_count = 0
    ou_sigma = OU_SIGMA

    # Establish the connection
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.bind(get_address(config))

    state_old = np.zeros(actor.s_dim)
    computed_action = np.zeros(ACTION_DIMS)
    check = False

    while True:

        ep_reward = 0
        terminal = 0

        while True:

            OriginalHandler = signal.signal(signal.SIGALRM, timeout_handler)

            # Receive the state from zeromq, the current state
            signal.alarm(30)
            try:
                incoming_message = server.recv()
            except TimeoutException:
                print ("No state received from GRL")
                code.kill()
                return

            # Get the length of the message
            len_incoming_message = len(incoming_message)

            # Decide which method sent the message and extract the message in a numpy array
            if len_incoming_message == (OBSERVATION_DIMS + 1) * 8:
                a = np.asarray(struct.unpack('d' * (OBSERVATION_DIMS + 1), incoming_message))
                test_agent = a[0]
                obs = a[1: OBSERVATION_DIMS + 1]
                state = np.array([math.cos(obs[0]), math.sin(obs[0]), obs[1]])
                episode_start = True
                episode_count += 1
                noise = np.zeros(actor.a_dim)
            else:
                a = np.asarray(struct.unpack('d' * (OBSERVATION_DIMS + 3), incoming_message))
                test_agent = a[0]
                obs = a[1: OBSERVATION_DIMS + 1]
                state = np.array([math.cos(obs[0]), math.sin(obs[0]), obs[1]])
                reward = a[OBSERVATION_DIMS + 1]
                terminal = a[OBSERVATION_DIMS + 2]
                episode_start = False

            # Add the transition to replay buffer
            if not episode_start:
                if test_agent:
                    if terminal == 2:
                        check = replay_buffer_test.add(np.reshape(state_old, (actor.s_dim,)),
                                          np.reshape(computed_action, (actor.a_dim,)),
                                          reward, True, np.reshape(state, (actor.s_dim,)))
                    else:
                        check = replay_buffer_test.add(np.reshape(state_old, (actor.s_dim,)),
                                          np.reshape(computed_action, (actor.a_dim,)),
                                          reward, False, np.reshape(state, (actor.s_dim,)))
                else:
                    if terminal == 2:
                        check = replay_buffer.add(np.reshape(state_old, (actor.s_dim,)),
                                          np.reshape(computed_action, (actor.a_dim,)),
                                          reward, True, np.reshape(state, (actor.s_dim,)))
                    else:
                        check = replay_buffer.add(np.reshape(state_old, (actor.s_dim,)),
                                          np.reshape(computed_action, (actor.a_dim,)),
                                          reward, False, np.reshape(state, (actor.s_dim,)))
            if check:
                print "Transitions saved"
                code.kill()
                return

            # Compute OU noise
            noise = ExplorationNoise.ou_noise(OU_THETA, OU_MU, ou_sigma, noise, ACTION_DIMS)

            # Compute action
            computed_action = compute_action(test_agent, actor, state, noise)

            state, computed_action = replay_buffer.sample_state_action(state, computed_action, episode_start)

            # Convert state to obs
            obs = invert(state)

            # Get state and action from replay buffer to send to GRL
            scaled_action = computed_action * ACTION_BOUND_REAL

            # Convert state and action into null terminated string
            outgoing_array = np.concatenate((scaled_action, obs))
            outgoing_message = struct.pack('d' * (ACTION_DIMS + OBSERVATION_DIMS), *outgoing_array)

            # Sends the predicted action via zeromq
            server.send(outgoing_message)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if not test_agent:
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

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

            # old_state = state

            state_old = state

            if not episode_start:
                ep_reward += reward

            if episode_start and episode_count != 1:
                print "Episode Ended:", ep_reward, episode_count
                if save_counter != 0 and episode_count % save_counter == 0 and episode_count != 0:
                    saver = tf.train.Saver()
                    saver.save(sess, "model-pendulum-{}.ckpt".format(1))
                break


def start(args):
    with tf.Session() as sess:
        # Initialize the actor and critic networks
        actor = ActorNetwork(sess, OBSERVATION_DIMS + 1, ACTION_DIMS, 1, \
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, OBSERVATION_DIMS + 1, ACTION_DIMS, CRITIC_LEARNING_RATE, TAU,
                               actor.get_num_trainable_vars())

        # Train the network
        train(args, sess, actor, critic)


