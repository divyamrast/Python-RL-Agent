#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pickle
from collections import deque
import numpy as np

transitions = deque()
with open('saved_data-1') as f:
    transitions = pickle.load(f)
    f.close()

    # batch = xrange(10000)
    s = np.array([_[0] for _ in transitions])
    a = np.array([_[1] for _ in transitions])
    s2 = np.array([_[4] for _ in transitions])
    b = np.concatenate((s,a,s2), axis=1)
    print b.shape
    np.savetxt('saved_data-1.csv',b,delimiter=',',newline='\n')

transitions = deque()
with open('saved_data-2') as f:
    transitions = pickle.load(f)
    f.close()
    s = np.array([_[0] for _ in transitions])
    a = np.array([_[1] for _ in transitions])
    s2 = np.array([_[4] for _ in transitions])
    b = np.concatenate((s,a,s2), axis=1)
    print b.shape
    np.savetxt('saved_data-2.csv',b,delimiter=',',newline='\n')