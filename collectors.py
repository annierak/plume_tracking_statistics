import numpy as np
import utility
import time

class TrackBoutCollector(object):

    def __init__(self,num_flies,wind_angle,source_pos,repeats=True):

        self.entry_distances = np.full(num_flies,np.nan)
        self.wind_angle = wind_angle
        self.source_pos = source_pos #shape of source_pos is 2 x traps
        self.failure_lengths = np.empty((2,0)) #will be an array whose columns contain [entry_distance,tracking_length]
        self.success_distances = np.empty(0) #will be a list of starting distances where the fly got to the trap
        self.passed_through_distances = np.full(num_flies,np.nan)

    #For now, whether we ignore flies that have previously tracked and lost
    #is decided outside of this object

    #In each of the below fly_ids is a boolean of length num_flies
    def update_for_detection(self,fly_ids,x_pos,y_pos,ever_tracked=None):

        entry_distances = self.compute_plume_distance(x_pos,y_pos)
        self.entry_distances[fly_ids] = entry_distances

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
            self.entry_distances[fly_ids] = np.nan

            # print(np.shape(self.failure_lengths))
            # print(np.shape(to_add))
            # raw_input()

    def update_for_trapped(self,fly_ids):

        self.success_distances = np.append(
            self.success_distances,self.entry_distances[fly_ids])

    def update_for_missed_detection(self,x_pos,y_pos,dispersal_mode_flies,ever_tracked):
        #Check for the flies whose y coordinate in trap coordinates is less than 0.17
            #(value chosen because at 1.6 m/s, dt=0.25, dispersal mode flies won't
            # ever remain in the same 0.1 x 0.1 m square for two consecutive time steps)

        #not_trapped is the bool of which flies are not trapped

        x,y = utility.shift_and_rotate(
                np.vstack((x_pos,y_pos)).T[:,:,np.newaxis],
                self.source_pos[np.newaxis,:,:],
                -self.wind_angle)

        #as of above line, shape of x and y is each (# flies) x (# sources)


        close_to_centerline_bool = (np.abs(y)<=0.17) & (x>0.)
        close_to_centerline_bool[~dispersal_mode_flies] = False  #exclude flies not in dispersal mode
        close_to_centerline_bool[ever_tracked] = False  #exclude flies which had tracked before
        close_to_centerline_xs = x[close_to_centerline_bool]
        # close_to_centerline_ys = y[close_to_centerline_bool]

        #collapse along trap axis
        close_to_centerline_bool = np.sum(close_to_centerline_bool,axis=1).astype(bool)

        # if np.sum(close_to_centerline_xs<20.)>0:
        #     print('---some flies passed thru---')
        #     print('at distances '+str(close_to_centerline_xs[close_to_centerline_xs<20.]))
        #     print(str(np.sqrt(np.square(
        #         close_to_centerline_xs[close_to_centerline_xs<20.])+
        #         np.square(close_to_centerline_ys[close_to_centerline_xs<20.]))))
        #     time.sleep(.1)

        #If they satisfy this value, add them (tentatively to the mask of flies that passed through)
        self.passed_through_distances[close_to_centerline_bool] = close_to_centerline_xs

        #Then, also check in with the mask of flies that detected odor (newly_surging) and cancel
        #out these indices to the list of passed through distances

        # self.passed_through_distances[newly_surging_ids] = np.nan

        return(close_to_centerline_bool)

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
