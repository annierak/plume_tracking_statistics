import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
import cPickle as pickle
import dill as pickle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from mpltools import color


import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
# import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors
sys.path.append("..")


#Wind angle
wind_angle = 0.
wind_mag = 1.6

#arena size
arena_size = 1000.


#Timing
dt = 0.25
plume_dt = 0.25
frame_rate = 20
times_real_time = 15 # seconds of simulation / sec in video
capture_interval = int(np.ceil(times_real_time*(1./frame_rate)/dt))


#    simulation_time = 20.*60. #seconds
simulation_time = 0.*60. #seconds
release_delay = 25.*60#/(wind_mag)
# release_delay = 5.

t_start = 0.0
t = 0. - release_delay


fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)


wind_param = {
                'speed': wind_mag,
                'angle': wind_angle,
                'evolving': False,
                'wind_dt': None,
                'dt': dt
                }
wind_field_noiseless = wind_models.WindField(param=wind_param)

#Setup ONE plume

#traps
trap_radius = 0.5
location_list = [(0,-100) ]
strength_list = [1]
trap_param = {
        'source_locations' : location_list,
        'source_strengths' : strength_list,
        'epsilon'          : 0.01,
        'trap_radius'      : trap_radius,
        'source_radius'    : 100
}

traps = trap_models.TrapModel(trap_param)

#Wind and plume objects

#Odor arena
xlim = (-arena_size, arena_size)
ylim = (-arena_size, arena_size)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
xlim[1]*1.2,ylim[1]*1.2)

source_pos = np.array([np.array(tup) for tup in traps.param['source_locations']]).T

#wind model setup
diff_eq = False
constant_wind_angle = wind_angle
aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
noise_gain=3.
noise_damp=0.071
noise_bandwidth=0.71
wind_grid_density = 200
Kx = Ky = 10000 #highest value observed to not cause explosion: 10000
wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
diff_eq=diff_eq,angle=constant_wind_angle,mag=wind_mag)


# Set up plume model
centre_rel_diff_scale = 2.
puff_release_rate = 10
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=int(2e5)
# max_num_puffs=100

plume_model = models.PlumeModel(
    sim_region, source_pos, wind_field,simulation_time+release_delay,
    plume_dt,plume_cutoff_radius=1500,
    centre_rel_diff_scale=centre_rel_diff_scale,
    puff_release_rate=puff_release_rate,
    puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
    max_num_puffs=max_num_puffs,max_distance_from_trap = 5000)

# Create a concentration array generator
array_z = 0.01

array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.
array_gen = processors.ConcentrationArrayGenerator(
    sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)



#Initial concentration plotting
conc_array = array_gen.generate_single_array(plume_model.puffs)
xmin = sim_region.x_min; xmax = sim_region.x_max
ymin = sim_region.y_min; ymax = sim_region.y_max
im_extents = (xmin,xmax,ymin,ymax)
vmin,vmax = 0.,50.
cmap = matplotlib.colors.ListedColormap(['white', 'orange'])


conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
vmin=vmin, vmax=vmax, cmap=cmap)

xmin,xmax,ymin,ymax = -arena_size,arena_size,-arena_size,arena_size

buffr = 50
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))

# ax.set_xlim((-200,200))
# ax.set_ylim((-200,200))


#Plot traps
for x,y in traps.param['source_locations']:
    plt.scatter(x,y,marker='x',s=50,c='k')
    p = matplotlib.patches.Circle((x, y), trap_radius,color='red',fill=False)
    ax.add_patch(p)

try:
    with open('single_saved_plume_orthogonal.pkl','r') as f:
        plume_model = pickle.load(f)

except(IOError):
    while t<simulation_time:
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))
            #update the swarm
            for j in range(int(dt/plume_dt)):
                wind_field.update(plume_dt)
                plume_model.update(plume_dt,verbose=True)
            t+=dt

conc_array = array_gen.generate_single_array(plume_model.puffs)
log_im = np.log(conc_array.T[::-1])
cutoff_l = np.percentile(log_im[~np.isinf(log_im)],1)
cutoff_u = np.percentile(log_im[~np.isinf(log_im)],99)

conc_im.set_data(log_im)
n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
conc_im.set_norm(n)

# num_traps x (1.1 x simulation_time x release rate) x 4 (x,y,z,r^2)
x_position = plume_model.puffs[:,:,0]
r_sqs = plume_model.puffs[:,:,3]

plt.figure()
plt.scatter(x_position,np.sqrt(r_sqs),alpha=0.5,label=)


with open('single_saved_plume_orthogonal.pkl','w') as f:
    pickle.dump(plume_model,f)


plt.show()
