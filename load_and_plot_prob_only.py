import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1], 'r') as f:
    (swarm_param,collector) = pickle.load(f)

num_flies = swarm_param['swarm_size']
#This doesn't work because the start position variables get overridden when the swarm is updated.
# fly_line_min = np.array([np.min(swarm_param['x_start_position']),np.min(swarm_param['y_start_position'])])
# fly_line_max = np.array([np.max(swarm_param['x_start_position']),np.max(swarm_param['y_start_position'])])
# print(fly_line_min,fly_line_max)
# fly_release_line_len = np.sqrt(np.sum(np.square(fly_line_max-fly_line_min)))
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

fig_save_name = 'distance_by_success_prob_histogram_'+sys.argv[1][:-4]

success_entry_distances = collector.success_distances

success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

bins=np.linspace(0,max_trap_distance,num_bins)

plt.figure(figsize=(14,7))
plt.subplot(211)
plt.xlim((0,max_trap_distance))
n_successes,_,_ = plt.hist(success_entry_distances,bins,alpha=0.5,color='b',histtype='step')

plt.xlabel('Distance from Trap')
plt.ylabel('Number Tracked to Source')

plt.legend()

plt.subplot(212)
plt.xlim((0,max_trap_distance))

plt.plot(bins[:-1],n_successes/(fly_release_density_per_bin),'o  ')

plt.xlim(0,max_trap_distance)
plt.xlabel('Distance from Trap')
plt.ylabel('Trap Arrival Probability')

plt.savefig(fig_save_name+'.png',format='png')




plt.show()
