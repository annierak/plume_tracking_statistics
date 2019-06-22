import numpy as np
import utility
import time
import odor_tracking_sim.swarm_models as swarm_models


class TrackBoutCollector(object):

    def __init__(self,num_flies,wind_angle,source_pos,repeats=True):

        self.entry_distances = np.full(num_flies,np.nan)
        self.wind_angle = wind_angle
        self.source_pos = source_pos #shape of source_pos is 2 x traps
        self.failure_lengths = np.empty((2,0)) #will be an array whose columns contain [entry_distance,tracking_length]
        self.success_distances = np.empty(0) #will be a list of starting distances where the fly got to the trap
        self.passed_through_distances = np.full(num_flies,np.nan)
        self.didnt_succeed_first_time_distances = np.full(num_flies,np.nan)

        self.repeats=repeats
        #Currently, setting repeats to False makes it such that once a fly
        #has crossed the plume centerline, it is excluded from being
        #counted as detecting the plume later on if it re-encounters it.

        #Currently, excluding flies that have already tracked is implemented
        #outside the object by anding with ~ever_tracked before passing to
        #update_for_detection

    #For now, whether we ignore flies that have previously tracked and lost
    #is decided outside of this object

    #In each of the below fly_ids is a boolean of length num_flies
    def update_for_detection(self,fly_ids,x_pos,y_pos,ever_tracked=None):

        # entry_distances = self.compute_plume_distance(x_pos,y_pos)
        entry_distances = self.compute_plume_distance(np.copy(x_pos),np.copy(y_pos))


        self.entry_distances[fly_ids] = entry_distances

        #If a given fly has a non-nan passed_through_distances value, its doesn't
        #get to be counted as a detection, so its entry_distances is reset to nan.

        self.entry_distances[~np.isnan(self.passed_through_distances)] = np.nan

    def update_for_loss(self,fly_ids,x_pos,y_pos):

        exit_distances = self.compute_plume_distance(x_pos,y_pos)

        to_add = np.vstack(
            (self.entry_distances[fly_ids],
                exit_distances-self.entry_distances[fly_ids]))
        if np.sum(fly_ids)>0:

            if np.size(self.failure_lengths)>0:
                self.failure_lengths = np.append(self.failure_lengths,
                    to_add,axis=1)
            else:
                self.failure_lengths = to_add

            #also update the didnt_succeed_first_time_distances variable
            fly_ids = fly_ids & (np.isnan(self.didnt_succeed_first_time_distances))
            self.didnt_succeed_first_time_distances[fly_ids] =  self.entry_distances[fly_ids]
            self.entry_distances[fly_ids] = np.nan

    def update_for_trapped(self,fly_ids):

        self.success_distances = np.append(
            self.success_distances,self.entry_distances[fly_ids])

    def update_for_missed_detection(self,x_pos,y_pos,dispersal_mode_flies,ever_tracked):
        #Check for the flies whose y coordinate in trap coordinates is less than 0.3
            #(value chosen because at 1.6 m/s, dt=0.25, dispersal mode flies will definitely
            # cross any bin of width 0.41/sqrt(2), but never remain in this bin
            #for two consecutive time steps)

        #not_trapped is the bool of which flies are not trapped

        x,y = utility.shift_and_rotate(
                np.vstack((x_pos,y_pos)).T[:,:,np.newaxis],
                self.source_pos[np.newaxis,:,:],
                -self.wind_angle)

        #as of above line, shape of x and y is each (# flies) x (# sources)


        # cutoff_distance = 0.17
        cutoff_distance = 0.3
        close_to_centerline_bool = (np.abs(y)<=cutoff_distance) #& (x>0.)

        close_to_centerline_bool[~dispersal_mode_flies] = False  #exclude flies not in dispersal mode
        close_to_centerline_bool[ever_tracked] = False  #exclude flies which had tracked before
        close_to_centerline_xs = x[close_to_centerline_bool]
        # close_to_centerline_ys = y[close_to_centerline_bool]

        #collapse along trap axis
        close_to_centerline_bool = np.sum(close_to_centerline_bool,axis=1).astype(bool)

        #If they satisfy this value, add them (tentatively to the mask of flies that passed through)
        self.passed_through_distances[close_to_centerline_bool] = close_to_centerline_xs

        #also update the didnt_succeed_first_time_distances variable
        self.didnt_succeed_first_time_distances[close_to_centerline_bool] = close_to_centerline_xs

        #Then, also check in with the mask of flies that detected odor (newly_surging) and cancel
        #out these indices to the list of passed through distances

        # self.passed_through_distances[newly_surging_ids] = np.nan

        # return(close_to_centerline_bool)


    def compute_plume_distance(self,x_pos,y_pos):
        # if np.shape(self.source_pos)[1]>1: #Case where there is more than one plume

        x,y = utility.shift_and_rotate(
                np.vstack((x_pos,y_pos)).T[:,:,np.newaxis],
                self.source_pos[np.newaxis,:,:],
                -self.wind_angle)

        #as of above line, shape of x and y is each (# flies) x (# sources)

        #For each fly, we assume it is chasing the plume whose source is closest
        # to it but where the coordinates in that plume's coordinate system is positive
        x[x<0.] = np.inf
        y[x<0.] = np.inf #this takes out of the running sources they are upwind of

        # closest_plume_x_inds,closest_plume_y_inds = np.where(x.T==np.min(x.T,axis=0))
        # closest_plume_xs = x.T[closest_plume_x_inds,closest_plume_y_inds]
        # closest_plume_ys = y.T[closest_plume_x_inds,closest_plume_y_inds]

        #now we have for each fly their distance to the plume source they are closest to.

        #for now, just use the x distance, can change later
        closest_plume_xs = np.min(x,axis=1)

        # return np.unique(closest_plume_xs) #uniqueifer might cause indexing probs if we add in y coords
        return closest_plume_xs

        # else: #Case where there is a single plume


class FlyCategorizer(object):
    '''Over the course of a simulation (1 plume simulation), track the
    fates of each fly:
    (-1) passed through without detecting
    (np.nan) tracked but never found source
    (0) successfully found source on 1st tracking bout
    (1) successfully found source on 2nd tracking bout
    ...


    Works in collaboration with (depends on) a TrackBoutCollector object which is
    simultaneously observing the swarm object.

    '''

    def __init__(self,num_flies):

        self.fate_vector = np.full(num_flies,np.nan)
        self.num_tracking_bouts = np.full(num_flies,0.)
        self.fate_assigned_mask = np.full(num_flies,False,dtype=bool)

    def update(self,swarm,collector,newly_trapped,newly_dispersing):


        #assign the flies who passed straight through to the failure fate (fate_vector)
        have_passed_through_bool = ~np.isnan(collector.passed_through_distances)
        # print('--------')
        # print(np.sum(have_passed_through_bool))
        # print('--------')
        # time.sleep(0.02)
        mask = have_passed_through_bool & (~self.fate_assigned_mask)
        # print(np.sum(mask))

        self.fate_vector[mask] = -1.

        # if np.sum(have_passed_through_bool)>0:
        #     print(np.unique(self.fate_vector))
        #     raw_input()


        #assign the flies who arrived at the trap to the success fate *WITH*
        #the number of tracking bouts

        self.fate_vector[newly_trapped]  = self.num_tracking_bouts[newly_trapped]


        #update the fated vs unfated screen
        self.fate_assigned_mask = self.fate_assigned_mask | mask | newly_trapped

        # print('----------')
        # print(np.sum(self.fate_assigned_mask))
        #
        # print('____________')
        # print(np.sum(self.fate_vector<0.))
        # #This value is going to be a little different than
        #np.sum(have_passed_through_bool) because np.sum(have_passed_through_bool)
        #includes trapped flies

        #for flies switching from casting to dispersing, add 1 to their tracking bout counts
        self.num_tracking_bouts[newly_dispersing] += 1.
        #
        # if (np.sum(have_passed_through_bool)>0) and (np.sum(self.fate_vector<0.)<1.):
        #     print(np.sum(self.fate_vector<0.))
        #     raw_input()
