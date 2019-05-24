import numpy as np

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

#Run from inside the folder for the behavior parameter of interest

behavior_param = 'detection_threshold'
cols = [behavior_param,'K','B','x_0','c']
# df = pd.DataFrame(columns=cols)

detection_thresholds = [0.05,0.075,0.125,0.15,0.175,0.2,0.225]

behavior_params = ['cast_interval',
            'odor_thresholds',
            'cast_timeout',
            'low_pass_filter_length']

file_name1='1m_uniform_release_times_'

rows_list = []
for val in detection_thresholds:
    param_spec_string = 'detection_threshold_'+str(val)
    file_name = file_name1 + param_spec_string
    output_file = file_name + '_logistic_fit_params.pkl'
    dict1 = {}
    with open(output_file, 'r') as f:
        (params_dict,swarm_param) = pickle.load(f)
    dict1.update({behavior_param:val})
    dict1.update(params_dict)
    dict1.update(dict(zip(behavior_params,[swarm_param[key] for key in behavior_params])))
    rows_list.append(dict1)
df = pd.DataFrame(rows_list)
print(df)
