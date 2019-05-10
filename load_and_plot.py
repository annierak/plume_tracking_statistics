import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1], 'r') as f:
    collector = pickle.load(f)

max_trap_distance = 1000
num_bins = 100

fig_save_name = 'distance_by_success_histogram_'+sys.argv[1][:-4]

#Display the collector.failure_lengths, collector.success_distances

failure_entry_distances = collector.failure_lengths[0,:]
success_entry_distances = collector.success_distances

failure_entry_distances = failure_entry_distances[~np.isnan(failure_entry_distances)]
failure_entry_distances = failure_entry_distances[~np.isinf(failure_entry_distances)]
success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]
try:
    passed_through_distances = collector.passed_through_distances[~np.isnan(collector.passed_through_distances)]
except:
    pass

bins=np.linspace(0,max_trap_distance,num_bins)

plt.figure(figsize=(14,7))
plt.subplot(211)
plt.xlim((0,max_trap_distance))
n_failures,bins,_ = plt.hist(failure_entry_distances,bins,alpha=0.5,label='Lost Plume',color='r',histtype='step')
n_successes,_,_ = plt.hist(success_entry_distances,bins,alpha=0.5,label='Tracked to Source',color='b',histtype='step')
try:
    n_passed,_,_ = plt.hist(passed_through_distances,bins,alpha=0.5,label='Passed Thru Plume',color='k',histtype='step')
except:
    pass

plt.xlabel('Distance from Trap')
plt.ylabel('Counts')

plt.legend()

plt.subplot(212)
plt.xlim((0,max_trap_distance))

try:
    plt.plot(bins[:-1],n_successes/(n_failures+n_successes+n_passed),'o  ')
except:
    plt.plot(bins[:-1],n_successes/(n_failures+n_successes),'o  ')
plt.xlim(0,max_trap_distance)
plt.xlabel('Distance from Trap')
plt.ylabel('Trap Arrival Probability')

plt.savefig(fig_save_name+'.png',format='png')




plt.show()
