import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import sys

from multiprocessing import Pool


def f(detection_thresholds):

    file_name1='1m_uniform_release_times_'


    num_flies = 1e6
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

    # fig_save_name = 'distance_by_success_prob_histogram_'+file_name

    plt.figure(figsize=(10,4))
    ax = plt.subplot()
    plt.xlim((0,max_trap_distance))

    # fig_save_name = 'single_plot_prob_histogram_across_'+'detection_thresholds'
    fig_save_name = 'single_plot_prob_histogram_across_'+'cast_timeouts'

    cmap = matplotlib.cm.get_cmap('copper')

    vals = np.linspace(0.,1.,len(detection_thresholds))
    colors = [cmap(val) for val in vals]

    for detection_threshold,color in zip(detection_thresholds,colors):


        # param_spec_string = 'detection_threshold_'+str(detection_threshold)
        param_spec_string = 'cast_timeout_'+str(detection_threshold)
        file_name = file_name1 + param_spec_string
        output_file = file_name + '.pkl'

        with open(output_file, 'r') as f:
            (swarm_param,collector) = pickle.load(f)
        success_entry_distances = collector.success_distances
        success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
        success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

        bins=np.linspace(0,max_trap_distance,num_bins)

        n_successes,_ = np.histogram(success_entry_distances,bins)
        plt.plot(bins[:-1],n_successes/(fly_release_density_per_bin),color=color,label=detection_threshold)

    sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=plt.Normalize(vmin=0.1, vmax=0.7))
    sm._A = []
    cb = plt.colorbar(sm,ticks=vals[::2])
    cb.ax.set_yticklabels(detection_thresholds[::2])
    # cb.set_label('Detection\nThreshold',rotation=0,labelpad=40,)
    cb.set_label('Cast\nTimeout',rotation=0,labelpad=40,)

    # plt.title('Trap-Finding Success as a Function of Plume Encounter Distance')
    plt.xlim(0,max_trap_distance)
    plt.ylim(-0.02,0.5)
    plt.xlabel('Distance from Trap (m)')
    plt.ylabel('Trap Arrival Probability')

    # plt.show()
    plt.savefig(fig_save_name+'.png',format='png')

# detection_thresholds = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225]
# detection_thresholds = [0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225]
# f(detection_thresholds)

cast_timeouts=[1,10,15,40,60,100]
f(cast_timeouts)
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
#
# cast_delays = [0.5,3,5,10,20,40]
# pool.map(f,cast_delays)

# cast_intervals= [[0.5,1.5],
#     [4,12],
#     [20,60]]

# cast_intervals= [[1,3],
#     [2,6],
#     [8,24],
#     [10,30]]
# pool.map(f,cast_intervals)
#
# iters = [1,2,3,4]
# pool.map(f,iters)
