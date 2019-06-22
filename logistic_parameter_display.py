import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import matplotlib.gridspec

cast_delays = [0.5,3,5,10]
cast_intervals = [
[0.5,1.5],
[1,3],
[2,6]
]
cast_timeouts = [10,15,40,60]
detection_thresholds = [0.05,0.075,0.1,0.125,0.15,0.2]

parameters = [cast_delays,cast_intervals,cast_timeouts,detection_thresholds]

parameter_strings = ['cast_delay','cast_interval','cast_timeout','detection_threshold']
logistic_parameter_strings = ['K','B','x_0']

fig=plt.figure(figsize=(6,8))

n_rows = len(parameter_strings);n_cols = len(logistic_parameter_strings)
gs = matplotlib.gridspec.GridSpec(nrows=n_rows, ncols=n_cols)


K_lims = [np.inf,-np.inf]
B_lims = [np.inf,-np.inf]
x_0_lims = [np.inf,-np.inf]

K_axes = []
B_axes = []
x_0_axes = []

for i,parameter in enumerate(parameters):

    parameter_string=parameter_strings[i]
    ks = []
    Bs = []
    x_0s = []


    for val in parameter:
        file_name = '1m_uniform_release_times_'+parameter_string+'_'+str(val)+'_logistic3_fit_params.pkl'
        with open(file_name,'r') as f:
            (params_dict,swarm_param) = pickle.load(f)
            ks.append(params_dict['K'])
            Bs.append(params_dict['B'])
            x_0s.append(params_dict['x_0'])

    K_lims[0]=np.min([K_lims[0],min(ks)])
    K_lims[1]=np.max([K_lims[1],max(ks)])
    B_lims[0]=np.min([B_lims[0],min(Bs)])
    B_lims[1]=np.max([B_lims[1],max(Bs)])
    x_0_lims[0]=np.min([x_0_lims[0],min(x_0s)])
    x_0_lims[1]=np.max([x_0_lims[1],max(x_0s)])

    for (j,logistic_parameter),p_axis in zip(enumerate([ks,Bs,x_0s]),[K_axes,B_axes,x_0_axes]):
        ax = fig.add_subplot(gs[i,j])
        p_axis.append(ax)
        ax.set_xlabel(parameter_string)
        ax.set_ylabel(logistic_parameter_strings[j])
        ax.plot(parameter,logistic_parameter,'o ')
        # plt.show()

for col,p_axis in enumerate([K_axes,B_axes,x_0_axes]):
    col_min = min([ax.get_ylim()[0] for ax in p_axis])
    col_max = max([ax.get_ylim()[1] for ax in p_axis])
    for row in range(4):
        p_axis[row].set_ylim(col_min,col_max)
        p_axis[row].set_aspect(1./p_axis[row].get_data_ratio())



fig.tight_layout()

plt.savefig('logistic_parameter_display_take1_samerange',format='png')
plt.show()
