"""
Package containing different types of exploration noise:
-   White noise
-   Ornstein-Uhlenbeck process
-   Noise decay
Original Author: Bart Keulen

Author: Divyam Rastogi
"""

import numpy as np


class ExplorationNoise(object):

    # ================================
    #    WHITE NOISE PROCESS
    # ================================
    @staticmethod
    def white_noise(mu, sigma, num_steps):
        # Generate random noise with mean 0 and variance 1
        return np.random.normal(mu, sigma, num_steps)

    # ================================
    #    ORNSTEIN-UHLENBECK PROCESS
    # ================================
    @staticmethod
    def ou_noise(theta, mu, sigma, num_steps, dt = 0.05):
        noise = np.ones(num_steps) * np.random.randn(1)

        # Solve using Euler-Maruyama method
        for i in xrange(1, num_steps):
            noise[i] = noise[i - 1] + theta * (mu - noise[i - 1]) * \
                                                dt + sigma * np.sqrt(dt) * np.random.randn(1)

        return noise

    # ================================
    #    EXPONENTIAL NOISE DECAY
    # ================================
    @staticmethod
    def exp_decay(noise, decay_end):
        num_steps = noise.shape[0]
        # Check if decay ends before end of noise sequence
        assert(decay_end <= num_steps)

        scaling = np.zeros(num_steps)

        scaling[:decay_end] = 2. - np.exp(np.divide(np.linspace(1., decay_end, num=decay_end) * np.log(2.), decay_end))

        return np.multiply(noise, scaling)

    # ================================
    #    TANH NOISE DECAY
    # ================================
    @staticmethod
    def tanh_decay(noise, decay_start, decay_length):
        num_steps = noise.shape[0]
        # Check if decay ends before end of noise sequence
        assert(decay_start + decay_length <= num_steps)

        scaling = 0.5*(1. - np.tanh(4. / decay_length * np.subtract(np.linspace(1., num_steps, num_steps),
                                                              decay_start + decay_length/2.)))

        return np.multiply(noise, scaling)