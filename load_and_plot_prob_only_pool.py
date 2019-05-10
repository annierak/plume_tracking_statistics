import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from multiprocessing import Pool


def f(detection_threshold):

    file_name='1m_uniform_release_times_'

    # param_spec_string = 'detection_threshold_'+str(detection_threshold)
    #
    # param_spec_string = 'cast_timeout_'+str(detection_threshold)
    # title_string = 'Cast Timeout: '+str(detection_threshold)

    param_spec_string = 'cast_delay_'+str(detection_threshold)
    title_string = 'Cast Delay: '+str(detection_threshold)

    file_name = file_name + param_spec_string

    output_file = file_name + '.pkl'

    with open(output_file, 'r') as f:
        (swarm_param,collector) = pickle.load(f)

    num_flies = swarm_param['swarm_size']
    fly_release_line_len =  np.sqrt(2)*1150

    max_trap_distance = 1000
    num_bins = 100
    bin_width = max_trap_distance/num_bins
    fly_release_density = num_flies/fly_release_line_len
    fly_release_density_per_bin = fly_release_density*bin_width
    print('release line length: '+str(fly_release_line_len))
    print('release density: '+str(fly_release_density))
    print('release density per bin: '+str(fly_release_density_per_bin))
    print('bin width: '+str(bin_width))

    fig_save_name = 'distance_by_success_prob_histogram_'+file_name

    success_entry_distances = collector.success_distances

    success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
    success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

    bins=np.linspace(0,max_trap_distance,num_bins)

    plt.figure(figsize=(14,7))
    plt.subplot(211)
    plt.title(title_string)
    plt.xlim((0,max_trap_distance))
    n_successes,_,_ = plt.hist(success_entry_distances,bins,alpha=0.5,color='b',histtype='step')

    plt.xlabel('Distance from Trap')
    plt.ylabel('Number Tracked to Source')

    plt.legend()

    plt.subplot(212)
    plt.xlim((0,max_trap_distance))

    plt.plot(bins[:-1],n_successes/(fly_release_density_per_bin),'o  ')

    plt.xlim(0,max_trap_distance)
    plt.ylim(-0.02,0.5)
    plt.xlabel('Distance from Trap')
    plt.ylabel('Trap Arrival Probability')

    plt.savefig(fig_save_name+'.png',format='png')

pool = Pool(processes=6)

# detection_thresholds = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225]
# pool.map(f, detection_thresholds)
# f(0.025)

# cast_timeouts=[1,10,15,40,60,100]
# pool.map(f, cast_timeouts)

#
# cast_intervals= [[0.5,1.5],
#     [1,3],
#     [2,6],
#     [4,12],
#     [8,24],
#     [10,30],
#     [20,60]]
# pool.map(f, cast_intervals)

cast_delays = [0.5,3,5,10,20,40]
pool.map(f,cast_delays)
