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
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors

from collectors import TrackBoutCollector

from multiprocessing import Pool

#In this simplified version of line_start_two_plumes, we just track the
#interception distances of flies which successfully track the plume to the source
#the first time they encounter it. We don't count flies that successfully tracked
#the second or nth time they encountered the plume.

no_repeat_tracking = True

#Wind angle
wind_angle = 5*np.pi/4
wind_mag = 1.6

#arena size
arena_size = 1000.

#file info
file_name='1m_uniform_release_times'

cast_delay = 10.
file_name = file_name +'video_errorless_surging_cast_delay_'+str(cast_delay)


# file_name='for_viewing_purposes'
# file_name='debugging_zero_peak_3'
output_file = file_name+'.pkl'

#Timing
dt = 0.25
plume_dt = 0.25
frame_rate = 20
times_real_time = 30 # seconds of simulation / sec in video
capture_interval = int(np.ceil(times_real_time*(1./frame_rate)/dt))


simulation_time = 10.*60. #seconds
release_delay = 25.*60#/(wind_mag)

t_start = 0.0
t = 0. - release_delay

# Set up figure
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)

#Video
FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)


wind_param = {
                'speed': wind_mag,
                'angle': wind_angle,
                'evolving': False,
                'wind_dt': None,
                'dt': dt
                }
wind_field_noiseless = wind_models.WindField(param=wind_param)

#Setup two plumes, locations (0,100) and (100,0)

#traps
trap_radius = 0.5
location_list = [(0,100) , (100,0)]
strength_list = [1,1]
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
# puff_release_rate = 0.001
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



#Start a bunch of flies with uniformly random headings at (0,0)
wind_slippage = (0.,0.)
# swarm_size=20000
# swarm_size=200000
swarm_size=1000000
# swarm_size=2000

release_times = scipy.random.uniform(0,simulation_time/2,size=swarm_size)

swarm_param = {
            'swarm_size'          : swarm_size,
            'heading_data'        : None,
            'initial_heading'     : np.radians(np.random.uniform(0.0,360.0,(swarm_size,))),
            'x_start_position'    : np.linspace(-arena_size,50,swarm_size),
            'y_start_position'    : np.linspace(-arena_size,50,swarm_size),
            'flight_speed'        : np.full((swarm_size,), 1.5),
            'release_time'        : release_times,
            'release_delay'       : release_delay,
            'cast_interval'       : [1,3],#cast_interval,
            'wind_slippage'       : wind_slippage,
            'odor_thresholds'     : {
                'lower': 0.0005,
                'upper': 0.05
                },
            'schmitt_trigger':False,
            'low_pass_filter_length':3, #seconds
            'dt_plot': capture_interval*dt,
            't_stop':simulation_time,
            'cast_timeout': 100,
            'airspeed_saturation':False
            }

swarm = swarm_models.BasicSwarmOfFlies(wind_field_noiseless,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)

#Set up collector object

#Check that source_pos is the right size
# print(source_pos)
# raw_input()
collector = TrackBoutCollector(swarm_size,wind_angle,source_pos,repeats=False)

#Plotting

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

#Conc array gen to be used for the flies
sim_region_tuple = plume_model.sim_region.as_tuple()
box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]

#for the plume distance cutoff version, make sure this is at least 2x radius
box_min,box_max = -arena_size,arena_size

r_sq_max=20;epsilon=0.00001;N=1e6

array_gen_flies = processors.ConcentrationValueFastCalculator(
            box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)

#Initial fly plotting
#Sub-dictionary for color codes for the fly modes
Mode_StartMode = 0
Mode_FlyUpWind = 1
Mode_CastForOdor = 2
Mode_Trapped = 3

edgecolor_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'red',
Mode_Trapped :   'black'}

facecolor_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'white',
Mode_Trapped :   'black'}

fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
fly_dots = plt.scatter(swarm.x_position, swarm.y_position,
    edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)

#Plot traps
for x,y in traps.param['source_locations']:
    plt.scatter(x,y,marker='x',s=50,c='k')
    p = matplotlib.patches.Circle((x, y), trap_radius,color='red',fill=False)
    ax.add_patch(p)


plt.ion()
plt.show()

#Start time loop
while t<simulation_time:
    for k in range(capture_interval):

        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        for j in range(int(dt/plume_dt)):
            wind_field.update(plume_dt)
            plume_model.update(plume_dt,verbose=True)
        if t>0.:
            dispersing_last_step = (swarm.mode == Mode_StartMode)
            casting_last_step = (swarm.mode == Mode_CastForOdor)
            not_trapped_last_step = (swarm.mode != Mode_Trapped)
            ever_tracked_last_step = swarm.ever_tracked

            swarm.update(t,dt,wind_field_noiseless,array_gen_flies,traps,plumes=plume_model,
                pre_stored=False)

            #Update the collector object
            newly_surging = dispersing_last_step & (swarm.mode == Mode_FlyUpWind)
            newly_dispersing = casting_last_step & (swarm.mode == Mode_StartMode)
            newly_trapped = not_trapped_last_step & (swarm.mode == Mode_Trapped)

            dispersal_mode_flies = (swarm.mode == Mode_StartMode)

            collector.update_for_trapped(newly_trapped)
            collector.update_for_loss(
                newly_dispersing,swarm.x_position[newly_dispersing],swarm.y_position[newly_dispersing])
            if no_repeat_tracking:
                newly_surging = newly_surging & (~ever_tracked_last_step)
            collector.update_for_detection(
                newly_surging,swarm.x_position[newly_surging],swarm.y_position[newly_surging])

        t+= dt

    if t>0:
        # Plotting
        fly_dots.set_offsets(np.c_[swarm.x_position,swarm.y_position])
        fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
        fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
        fly_dots.set_edgecolor(fly_edgecolors)
        fly_dots.set_facecolor(fly_facecolors)


        if t<2.:
            conc_array = array_gen.generate_single_array(plume_model.puffs)

            log_im = np.log(conc_array.T[::-1])
            cutoff_l = np.percentile(log_im[~np.isinf(log_im)],1)
            cutoff_u = np.percentile(log_im[~np.isinf(log_im)],99)

            conc_im.set_data(log_im)
            n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
            conc_im.set_norm(n)


        plt.pause(0.001)
        writer.grab_frame()

writer.finish()


#Save the collector object with pickle
with open(output_file, 'w') as f:
    pickle.dump((swarm_param,collector),f)
