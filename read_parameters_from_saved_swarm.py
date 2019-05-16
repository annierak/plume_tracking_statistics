# Input the address of a saved swarm pkl, and load it and print out the
# parameter values used for this simulation.

import cPickle as pickle
import numpy as np
import sys


output_file = sys.argv[1]
with open(output_file, 'r') as f:
    (swarm_param,collector) = pickle.load(f)

print(output_file)

keys = ['surging_error_std',
            'cast_interval',
            'odor_thresholds',
            'cast_timeout',
            'low_pass_filter_length']


for key in keys:
    try:
        print(key,swarm_param[key])
    except(KeyError):
        print('swarm has no '+str(key))
