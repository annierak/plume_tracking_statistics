import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from multiprocessing import Pool


def f(duration):


    file_name='1m_uniform_release_times_default_params_'

    title_string = 'Default Parameters after '+str(duration)+' min'


    param_spec_string = 't_'+str(duration)+'min'


    file_name = file_name + param_spec_string
    output_file = file_name + '.pkl'

    fig_save_name = 'two_panel_histogram_'+file_name

    with open(output_file, 'r') as f:
        (swarm_param,collector) = pickle.load(f)

    num_flies = swarm_param['swarm_size']*(duration/20.)
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


    success_entry_distances = collector.success_distances
    success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
    success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

    failure_entry_distances = collector.failure_lengths[0,:]
    failure_entry_distances = failure_entry_distances[~np.isnan(failure_entry_distances)]
    failure_entry_distances = failure_entry_distances[~np.isinf(failure_entry_distances)]


    bins=np.linspace(0,max_trap_distance,num_bins)

    plt.figure(figsize=(10,4))


    plt.subplot(211)
    plt.xlim((0,max_trap_distance))
    plt.ylim((0,6000))
    n_failures,bins,_ = plt.hist(failure_entry_distances,bins,alpha=0.5,label='Lost Plume',color='r',histtype='step')
    n_successes,_,_ = plt.hist(success_entry_distances,bins,alpha=0.5,label='Tracked to Source',color='b',histtype='step')
    try:
        n_passed,_,_ = plt.hist(passed_through_distances,bins,alpha=0.5,label='Passed Thru Plume',color='k',histtype='step')
    except:
        pass

    plt.xlabel('Distance from Trap')
    plt.ylabel('Counts')

    plt.legend()
    plt.title(title_string,color='purple')

    plt.subplot(212)


    plt.xlim((0,max_trap_distance))
    n_successes,_ = np.histogram(success_entry_distances,bins)#,alpha=0.5,color='b',histtype='step')
    plt.plot(bins[:-1],n_successes/(fly_release_density_per_bin),'o  ')

    plt.xlim(0,max_trap_distance)
    plt.ylim(-0.02,0.5)
    plt.xlabel('Distance from Trap (m)')
    plt.ylabel('Trap Arrival Probability')

    plt.savefig(fig_save_name+'.png',format='png')

pool = Pool(processes=6)

durations = [2.0,4.0,6.0,8.0,10.0,12.0]#,14.0,16.0,18.0]#,20.0]
pool.map(f,durations)
