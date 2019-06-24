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
import matplotlib.cm

import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
# import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors
sys.path.append("..")
from matplotlib.widgets import Slider

detection_threshold = 0.05

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
y_position = plume_model.puffs[:,:,1]
r_sqs = plume_model.puffs[:,:,3]

x_position = x_position[~np.isnan(x_position)]
y_position = y_position[~np.isnan(y_position)]
r_sqs = r_sqs[~np.isnan(r_sqs)]

puff_mol_amt =1.
big_K = (puff_mol_amt / (8 * np.pi**3)**0.5) / r_sqs**1.5
threshold_radii = np.sqrt(-1*np.log(detection_threshold/(big_K))*2*r_sqs)

# 
# plt.figure()
# bins = np.linspace(0,1000,100)
# plt.hist(x_position,bins = bins)
# plt.show()

# plt.figure()
# plt.scatter(x_position,r_sqs,alpha=0.5)

fig = plt.figure()
ax = plt.subplot(3,1,1)
# plt.scatter(x_position,y_position,alpha=0.)
plt.xlim([0,1000])
# ppd=72./ax.figure.dpi
trans = ax.transData.transform
# s =  ((trans((1,1))-trans((0,0)))*ppd)[1]
s =  ((trans((1,1))-trans((0,0))))[1]
s =  1./50.
# s = ((ax.get_window_extent().width  / (vmax-vmin+1.) * 72./fig.dpi) ** 2)


cmap = matplotlib.cm.get_cmap('winter')
normalized_r_sqs = r_sqs/(np.max(r_sqs))
# puff_dots = plt.scatter(x_position,y_position,alpha=0.5,s=r_sqs,
    # facecolor=cmap(normalized_r_sqs),edgecolor=cmap(normalized_r_sqs))
# plt.colorbar()


puff_dots = plt.scatter(x_position,y_position,alpha=0.5,
    s=r_sqs,
    c=r_sqs,
    edgecolor = None,
    vmin=0.,vmax=np.max(r_sqs[x_position<700.]),cmap=cmap)
plt.colorbar()

threshold_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
threshold_slider = Slider(threshold_ax,
    'Detection Threshold', 0.01, 0.25, valinit=0.05)#, valstep=0.01)

fig2 = plt.figure(10)
r_sq_inputs = np.linspace(0.01,8.,100)
bin_counts,_ = np.histogram(threshold_radii,bins=r_sq_inputs)
print(np.shape(bin_counts))
print(np.shape(r_sq_inputs))
counts, = plt.plot(r_sq_inputs[1:],bin_counts)
plt.xlabel('r_sq')
plt.ylabel('Threshold Radius')

fig3 = plt.figure()
thres_rad_plot, = plt.plot(x_position,threshold_radii,'o')
plt.xlabel('x position')
plt.ylabel('Threshold Radius')
plt.xlim([0,1000])
plt.ylim([0,8])

fig4 = plt.figure()

box_min,box_max = -arena_size,arena_size
r_sq_max=20;epsilon=0.00001;N=1e6
array_gen_flies = processors.ConcentrationValueFastCalculator(
            box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)

x_inputs = np.linspace(1,1000,1000)
conc_values = array_gen_flies.calc_conc_list(plume_model.puffs, x_inputs, -100*np.ones_like(x_inputs))
plt.plot(x_inputs,conc_values)
plt.ylabel('Sampled Concentration')
plt.xlabel('x position')
plt.ylim([0,1])
thres_line = plt.axhline(detection_threshold)



def update(val):
    detection_threshold = threshold_slider.val
    threshold_radii = np.sqrt(-1*np.log(detection_threshold/(big_K))*2*r_sqs)
    sizes = np.zeros_like(x_position)
    sizes[~np.isnan(threshold_radii)] = threshold_radii[
        ~np.isnan(threshold_radii)]/s
    puff_dots.set_sizes(threshold_radii/s, dpi=72.0)

    bin_counts,_ = np.histogram(threshold_radii,bins=r_sq_inputs)
    # print(bin_counts)
    counts.set_ydata(bin_counts)

    thres_rad_plot.set_ydata(threshold_radii)
    thres_line.set_ydata(detection_threshold)

    fig.canvas.draw_idle()
    fig2.canvas.draw_idle()
    fig3.canvas.draw_idle()
    fig4.canvas.draw_idle()


threshold_slider.on_changed(update)


with open('single_saved_plume_orthogonal.pkl','w') as f:
    pickle.dump(plume_model,f)


plt.show()
