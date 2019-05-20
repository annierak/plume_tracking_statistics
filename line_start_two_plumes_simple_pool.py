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


def f(detection_threshold):
# def f(cast_timeout):
# def f(cast_delay):
# def f(cast_interval):

    no_repeat_tracking = True


    #Comment these out depending on which parameter we're iterating through
    # detection_threshold = 0.05
    cast_timeout = 20.
    cast_interval = [1,3]
    cast_delay = 3.

    #Wind angle
    wind_angle = 5*np.pi/4
    wind_mag = 1.6

    #arena size
    arena_size = 1000.

    #file info
    file_name='1m_uniform_release_times_'
    file_name = file_name +'detection_threshold_'+str(detection_threshold)
    # file_name = file_name +'cast_timeout_'+str(cast_timeout)
    # file_name = file_name +'cast_interval_'+str(cast_interval)
    # file_name = file_name +'cast_delay_'+str(cast_delay)
    # file_name = file_name +'errorless_surging_cast_delay_'+str(cast_delay)

    # file_name='for_viewing_purposes'
    # file_name='debugging_zero_peak_3'
    output_file = file_name+'.pkl'

    #Timing
    dt = 0.25
    plume_dt = 0.25
    frame_rate = 20
    times_real_time = 30 # seconds of simulation / sec in video
    capture_interval = int(np.ceil(times_real_time*(1./frame_rate)/dt))


#    simulation_time = 20.*60. #seconds
    simulation_time = 12.*60. #seconds
    release_delay = 25.*60#/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay

    # Set up figure
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

    #Set up collector object

    #Check that source_pos is the right size
    collector = TrackBoutCollector(swarm_size,wind_angle,source_pos,repeats=False)

    #Conc array gen to be used for the flies
    sim_region_tuple = plume_model.sim_region.as_tuple()
    box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]

    #for the plume distance cutoff version, make sure this is at least 2x radius
    box_min,box_max = -arena_size,arena_size

    r_sq_max=20;epsilon=0.00001;N=1e6

    array_gen_flies = processors.ConcentrationValueFastCalculator(
                box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)

    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3


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
                collector.update_for_missed_detection(swarm.x_position,swarm.y_position
                    ,dispersal_mode_flies,ever_tracked_last_step)


            t+= dt

    #Save the collector object with pickle
    with open(output_file, 'w') as f:
        swarm_param.update({'simulation_duration':t})
        pickle.dump((swarm_param,collector),f)

pool = Pool(processes=4)

#detection_thresholds = [0.025,0.075,0.1,0.125,0.15,0.175,0.2,0.225]
detection_thresholds = [0.025,0.05,0.075,0.1]#,0.125,0.15,0.175,0.2,0.225]
pool.map(f, detection_thresholds)

# cast_timeouts=[1,10,15,40,60,100]
# pool.map(f, cast_timeouts)

# cast_intervals= [[0.5,1.5],
#     [4,12],
#     [20,60]]
# cast_intervals1= [[1,3],
#     [2,6],
#     [8,24],
#     [10,30]]

# pool.map(f, cast_intervals)
# pool.map(f, cast_intervals1)

# cast_delays = [0.5,3,5,10,20,40]
# pool.map(f,cast_delays)
