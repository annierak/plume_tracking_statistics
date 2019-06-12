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
from mpltools import color


import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
# import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors
sys.path.append("..")
# from ... import collectors

from collectors import TrackBoutCollector

from multiprocessing import Pool

#In this simplified version, we just track the
#interception distances of flies which successfully track the plume to the source
#the first time they encounter it. We don't count flies that successfully tracked
#the second or nth time they encountered the plume.

detection_threshold = 0.05
# detection_threshold = 0.1

no_repeat_tracking = True


#Comment these out depending on which parameter we're iterating through
# detection_threshold = 0.05
cast_timeout = 20.
# cast_timeout = 40.
cast_interval = [1,3]
cast_delay = 3.

#Wind angle
wind_angle = 5*np.pi/4
wind_mag = 1.6

#arena size
arena_size = 1000.

#file info
file_name='two_histogram_video_1_minus_S'
output_file = file_name+'.pkl'

#Timing
dt = 0.25
plume_dt = 0.25
frame_rate = 20
times_real_time = 15 # seconds of simulation / sec in video
capture_interval = int(np.ceil(times_real_time*(1./frame_rate)/dt))


#    simulation_time = 20.*60. #seconds
simulation_time = 15.*60. #seconds
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
location_list = [(0,100) ]
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



#Start a bunch of flies with uniformly random headings at (0,0)
wind_slippage = (0.,0.)
# swarm_size=20000
# swarm_size=200000
# swarm_size=1000000
swarm_size=2000

# release_times = scipy.random.uniform(0,simulation_time/2,size=swarm_size)
release_times = np.zeros(shape=swarm_size)

swarm_param = {
            'swarm_size'          : swarm_size,
            'heading_data'        : None,
            'initial_heading'     : np.radians(np.full((swarm_size,),135.)), #for orthogonal departure
            'x_start_position'    : np.linspace(-arena_size,50,swarm_size),
            'y_start_position'    : np.linspace(-arena_size,50,swarm_size),
            'flight_speed'        : np.full((swarm_size,), 1.5),
            'release_time'        : release_times,
            'release_delay'       : release_delay,
            'cast_interval'       : cast_interval,
            'wind_slippage'       : wind_slippage,
            'odor_thresholds'     : {
                'lower': 0.0005,
                'upper': detection_threshold
                },
            'schmitt_trigger':False,
            'low_pass_filter_length':cast_delay, #seconds
            'dt_plot': capture_interval*dt,
            't_stop':simulation_time,
            'cast_timeout': cast_timeout,
            'surging_error_std'   : scipy.radians(1e-10),
            'airspeed_saturation':False
            }

swarm = swarm_models.BasicSwarmOfFlies(wind_field_noiseless,traps,param=swarm_param,
    start_type='fh',track_plume_bouts=False,track_arena_exits=False)

#Release density variables
num_bins = 100
max_trap_distance = 1000
bin_width = max_trap_distance/num_bins
fly_release_line_len = int(np.sqrt(
    (np.max(swarm.param['x_start_position'])-np.min(swarm.param['x_start_position']))**2+
    (np.max(swarm.param['y_start_position'])-np.min(swarm.param['y_start_position']))**2
        ))
fly_release_density = swarm_size/fly_release_line_len
fly_release_density_per_bin = fly_release_density*bin_width
bins=np.linspace(0,max_trap_distance,num_bins)

#Set up collector object

#Check that source_pos is the right size
collector = TrackBoutCollector(swarm_size,wind_angle,source_pos,repeats=False)


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



#Set up live histogram plot
fig2 = plt.figure(figsize=(10,11))
ax = plt.subplot(3,1,1)

#(1)--------
success_entry_distances = collector.success_distances
success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

n_successes,_ = np.histogram(success_entry_distances,bins)
curve1, = plt.plot(bins[:-1],n_successes/(fly_release_density_per_bin),'-o',
    label='Succeeded the first time after detecting',alpha=0.5)

#(2)--------
failure_entry_distances = collector.failure_lengths[0,:]
failure_entry_distances = failure_entry_distances[~np.isnan(failure_entry_distances)]
failure_entry_distances = failure_entry_distances[~np.isinf(failure_entry_distances)]

n_failures,_ = np.histogram(failure_entry_distances,bins)
curve2, = plt.plot(bins[:-1],n_failures/(fly_release_density_per_bin),'-o',
    label='Entrance point before failing',alpha=0.5)

#(6)--------
passed_through_distances = collector.passed_through_distances
passed_through_distances = passed_through_distances[~np.isnan(passed_through_distances)]
passed_through_distances = passed_through_distances[~np.isinf(passed_through_distances)]

n_passed,_ = np.histogram(passed_through_distances,bins)
curve6, = plt.plot(bins[:-1],n_passed/(fly_release_density_per_bin),'-o',
    label='Passed straight through here',alpha=0.5)

plt.xlim(0,max_trap_distance)
plt.xlabel('Distance from Trap (m)')
plt.ylabel('Fraction')
text = ax.text(0.2,1,'0 s',transform=ax.transAxes)
plt.legend(bbox_to_anchor=(1., 1.4))


ax2 = plt.subplot(3,1,2)
color.cycle_cmap(3, cmap='Dark2', ax=ax2)


#(3)--------
entry_distances = collector.entry_distances
entry_distances = entry_distances[~np.isnan(entry_distances)]
entry_distances = entry_distances[~np.isinf(entry_distances)]

n_entries,_ = np.histogram(entry_distances,bins)
curve3, = plt.plot(bins[:-1],n_entries/(fly_release_density_per_bin),'-o',
    label='Most recently entered here',alpha=0.5)


#(4)--------
surging_distances = collector.compute_plume_distance(
    swarm.x_position[swarm.mode == Mode_FlyUpWind],
    swarm.y_position[swarm.mode == Mode_FlyUpWind])

n_surging,_ = np.histogram(entry_distances,bins)
curve4, = plt.plot(bins[:-1],n_surging/(fly_release_density_per_bin),'-o',
    label='Currently surging here',alpha=0.5)

#(5)--------
casting_distances = collector.compute_plume_distance(
    swarm.x_position[swarm.mode == Mode_CastForOdor],
    swarm.y_position[swarm.mode == Mode_CastForOdor])

n_casting,_ = np.histogram(casting_distances,bins)
curve5, = plt.plot(bins[:-1],n_casting/(fly_release_density_per_bin),'-o',
    label='Currently casting here',alpha=0.5)

plt.xlim(0,max_trap_distance)
plt.xlabel('Distance from Trap (m)')
plt.ylabel('Fraction')
plt.legend(bbox_to_anchor=(1., -0.3))


#Video
FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig2, file_name+'.mp4', 500)



# plt.ion()
# plt.show()
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

            collector.update_for_missed_detection(swarm.x_position,swarm.y_position,
                dispersal_mode_flies,ever_tracked_last_step)
        t+= dt

    if t>0: #Histogram updating

        text.set_text(str(np.floor(t/60.))[0]+' min '+str(t%60.)[0:2]+' sec')

        #(1)--------
        success_entry_distances = collector.success_distances
        success_entry_distances = success_entry_distances[~np.isnan(success_entry_distances)]
        success_entry_distances = success_entry_distances[~np.isinf(success_entry_distances)]

        n_successes,_ = np.histogram(success_entry_distances,bins)
        curve1.set_ydata(n_successes.astype(float)/(fly_release_density_per_bin))

        max_n = max(n_successes)

        #(2)--------
        failure_entry_distances = collector.failure_lengths[0,:]
        failure_entry_distances = failure_entry_distances[~np.isnan(failure_entry_distances)]
        failure_entry_distances = failure_entry_distances[~np.isinf(failure_entry_distances)]

        n_failures,_ = np.histogram(failure_entry_distances,bins)
        curve2.set_ydata(n_failures.astype(float)/(fly_release_density_per_bin))
        max_n = max(max_n,max(n_failures))

        #(6)--------
        passed_through_distances = collector.passed_through_distances
        passed_through_distances = passed_through_distances[~np.isnan(passed_through_distances)]
        passed_through_distances = passed_through_distances[~np.isinf(passed_through_distances)]

        n_passed,_ = np.histogram(passed_through_distances,bins)
        curve6.set_ydata(n_passed.astype(float)/(fly_release_density_per_bin))
        max_n = max(max_n,max(n_passed))



        #(3)--------
        entry_distances = collector.entry_distances
        entry_distances = entry_distances[~np.isnan(entry_distances)]
        entry_distances = entry_distances[~np.isinf(entry_distances)]

        n_entries,_ = np.histogram(entry_distances,bins)
        curve3.set_ydata(n_entries.astype(float)/(fly_release_density_per_bin))
        max_n = max(max_n,max(n_entries))

        #(4)--------
        surging_distances = collector.compute_plume_distance(
            swarm.x_position[swarm.mode == Mode_FlyUpWind],
            swarm.y_position[swarm.mode == Mode_FlyUpWind])

        surging_distances = surging_distances[~np.isnan(surging_distances)]
        surging_distances = surging_distances[~np.isinf(surging_distances)]

        n_surging,_ = np.histogram(surging_distances,bins)
        curve4.set_ydata(n_surging.astype(float)/(fly_release_density_per_bin))
        max_n = max(max_n,max(n_surging))

        #(5)--------
        casting_distances = collector.compute_plume_distance(
            swarm.x_position[swarm.mode == Mode_CastForOdor],
            swarm.y_position[swarm.mode == Mode_CastForOdor])
        casting_distances  = casting_distances[~np.isnan(casting_distances)]
        casting_distances  = casting_distances[~np.isinf(casting_distances)]

        n_casting,_ = np.histogram(casting_distances,bins)
        curve5.set_ydata(n_casting.astype(float)/(fly_release_density_per_bin))
        max_n = max(max_n,max(n_casting))


        ax.set_ylim([0,max_n/(fly_release_density_per_bin)])
        ax2.set_ylim([0,max_n/(fly_release_density_per_bin)])






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


        writer.grab_frame()

        # plt.pause(0.001)


#Save the collector object with pickle
with open(output_file, 'w') as f:
    swarm_param.update({'simulation_duration':t})
    pickle.dump((swarm_param,collector),f)

# pool = Pool(processes=4)

# detection_thresholds = [0.025,0.05, 0.075,0.1,0.125,0.15,0.175,0.2,0.225]
# f(0.05)
