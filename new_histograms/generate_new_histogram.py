import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from collectors import TrackBoutCollector,FlyCategorizer



# file_name = 'categorizer_object_method_small'
file_name='categorizer_method_1m_sustained_release'
output_file = file_name + '.pkl'

with open(output_file, 'r') as f:
    (swarm,swarm_param,collector,categorizer) = pickle.load(f)

num_flies = swarm_param['swarm_size']

fly_release_line_len = 1000


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
fig_save_name = 'single_plot_prob_histogram_'+file_name

#Use the categorizer object to find the inds of the flies in each category
# (-1) passed through without detecting
passed_through_inds = (categorizer.fate_vector<0.)

# (np.nan) tracked but never found source
tracked_but_failed_inds = np.isnan(categorizer.fate_vector)

# (2) successfully found source on 1st tracking bout
succeed_bout_0_inds = (categorizer.fate_vector == 0.)

# (3) successfully found source on 2nd tracking bout
succeed_bout_1_inds = (categorizer.fate_vector == 1.)

# (3) successfully found source on 3rd or greater tracking bout
succeed_bout_2_inds = (categorizer.fate_vector >= 2.)


#Map these inds to distances
passed_through_distances = swarm.param['x_start_position'][passed_through_inds]
tracked_but_failed_distances = swarm.param['x_start_position'][tracked_but_failed_inds]
succeed_bout_0_distances = swarm.param['x_start_position'][succeed_bout_0_inds]
succeed_bout_1_distances = swarm.param['x_start_position'][succeed_bout_1_inds]
succeed_bout_2_distances = swarm.param['x_start_position'][succeed_bout_2_inds]

all_distances = [succeed_bout_0_distances,
succeed_bout_1_distances,
succeed_bout_2_distances,
tracked_but_failed_distances,
passed_through_distances]

# bins=np.linspace(0,max_trap_distance,num_bins)
bins=np.arange(0.,1000.,1000./num_bins)

names = ['succeeded first time',
    'succeeded second time',
    'succeeded 3rd thru nth time',
    'tracked but never arrived',
    'never detected plume']

colors = [
    'red',
    'orange',
    'yellow',
    'purple',
    'black'
    ]


fig = plt.figure(figsize=(10,4))
ax = plt.subplot()
plt.xlim((0,max_trap_distance))
n_successes,bins,patches = plt.hist(all_distances,bins,
     histtype='bar',
         stacked=True,
         fill=True,
         label=names,
         alpha=0.8, # opacity of the bars
         color=colors)
# for row in patches:
#     for col in row:
#         col.set_height(col.get_height()/(fly_release_density_per_bin))


plt.xlim(0,max_trap_distance)
# plt.ylim(-0.02,0.5)
plt.ylim(0,fly_release_density_per_bin)
plt.xlabel('Distance from Trap (m)')
plt.ylabel('Fraction')
plt.legend()

fig.canvas.draw()

new_labels = [float(yticklabel.get_text())/(fly_release_density_per_bin)
    for yticklabel in ax.get_yticklabels()]
# print(new_labels)
ax.set_yticklabels(new_labels)

# plt.savefig(fig_save_name+'.png',format='png')
plt.savefig(fig_save_name+'_flipped.png',format='png')
plt.show()
