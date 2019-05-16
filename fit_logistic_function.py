import numpy as np

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

from multiprocessing import Pool


def logistic(x,K,B,x_0,v):
        Q = 1.
        return -K+K/((1.+Q*np.exp(-B*(x-x_0))**(1./v)))
def logistic3(x,K,B,x_0):
        Q = 1.
        return -K+K/((1.+Q*np.exp(-B*(x-x_0))))

def f(detection_threshold):

    file_name='1m_uniform_release_times_'

    param_spec_string = 'detection_threshold_'+str(detection_threshold)
    title_string = 'Detection Threshold: '+str(detection_threshold)

    # param_spec_string = 'cast_timeout_'+str(detection_threshold)
    # title_string = 'Cast Timeout: '+str(detection_threshold)

    # param_spec_string = 'cast_delay_'+str(detection_threshold)
    # title_string = 'Cast Delay: '+str(detection_threshold)

    # param_spec_string = 'errorless_surging_cast_delay_'+str(detection_threshold)
    # title_string = 'No Surging Error; Cast Delay: '+str(detection_threshold)

    # param_spec_string = 'cast_interval_'+str(detection_threshold)
    # title_string = 'Cast Interval: '+str(detection_threshold)

    # param_spec_string = 'default_params_iter_'+str(detection_threshold)
    # title_string = 'Default Params Iteration: '+str(detection_threshold)

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

    # fig_save_name = 'distance_by_success_prob_histogram_'+file_name
    fig_save_name = 'prob_logistic_fitting_'+file_name

    success_entry_distances = collector.success_distances

    success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
    success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

    bins=np.linspace(0,max_trap_distance,num_bins)

    plt.figure(figsize=(10,4))
    plt.xlim((0,max_trap_distance))
    n_successes,_,_ = plt.hist(success_entry_distances,bins,alpha=0.5,color='b',histtype='step')
    plt.clf()
    probs = n_successes/(fly_release_density_per_bin)
    plt.plot(bins[:-1],probs,'o  ')

    plt.title(title_string,color='purple')
    plt.xlim(0,max_trap_distance)
    plt.ylim(-0.02,0.5)
    plt.xlabel('Distance from Trap (m)')
    plt.ylabel('Trap Arrival Probability')


    #Fitting portion
    initial_guess = (-1.,3.,1.,0.5)

    #This bit checks that the fitting algorithm re-finds inputted parameters.
    # bins_dummy =  np.linspace(1,10,10000)
    # # probs_dummy = logistic(bins_dummy,-0.5,.1,6.,0.1)
    # probs_dummy = logistic3(bins_dummy,-0.5,2.,6.)
    # # p_opt,p_cov = curve_fit(logistic,bins_dummy,
    # #     probs_dummy,p0=initial_guess,bounds=([-np.inf,-np.inf,-np.inf,0],np.inf))
    # p_opt,p_cov = curve_fit(logistic3,bins_dummy,
    #     probs_dummy,p0=initial_guess[:-1])
    #
    # # K_est,B_est,x_0_est,v_est = p_opt
    # K_est,B_est,x_0_est = p_opt
    #
    # print(p_opt)
    # plt.figure(2)
    #
    # # prob_est = logistic(bins[:-1],K_est,B_est,x_0_est,v_est)
    # # prob_est = logistic(bins_dummy,K_est,B_est,x_0_est,v_est)
    # prob_est = logistic3(bins_dummy,K_est,B_est,x_0_est)
    # plt.plot(bins_dummy,prob_est,color='red',label='logistic fit')
    # plt.plot(bins_dummy,probs_dummy,'o ',markersize=.5)

    p_opt,p_cov = curve_fit(logistic3,bins[:-1],
        probs,p0=initial_guess[:-1])

    # K_est,B_est,x_0_est,v_est = p_opt
    K_est,B_est,x_0_est = p_opt
    prob_est = logistic3(bins[:-1],K_est,B_est,x_0_est)
    plt.plot(bins[:-1],prob_est,color='red',label='logistic fit')




    plt.legend()
    plt.show()

    plt.savefig(fig_save_name+'.png',format='png')


f(0.05)

# detection_thresholds = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225]
# pool = Pool(processes=6)
# pool.map(f, detection_thresholds)
