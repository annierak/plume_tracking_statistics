import numpy as np
import utility

class TrackBoutCollector(object):

    def __init__(self,num_flies,wind_angle,source_pos,):

        self.entry_distances = np.full(np.nan,shape=num_flies)
        self.wind_angle = wind_angle
        self.source_pos = source_pos #shape of source_pos is 2 x traps
        self.failure_lengths = np.empty((2,0)) #will be an array whose columns contain [entry_distance,tracking_length]
        self.success_distances = np.empty(0) #will be a list of starting distances where the fly got to the trap

    def update_for_detection(self,fly_ids,locations):

        entry_distances = f(locations)

        self.entry_distances[fly_ids] = entry_distances

    def update_for_loss(self,fly_ids,locations):

        exit_distances = f(locations)

        np.append(self.failure_lengths,
            np.vstack(
                (self.entry_distances[fly_ids],
                    exit_distances-self.entry_distances[fly_ids])))
        self.entry_distances[fly_ids] = np.nan

    def update_for_trapped(self,fly_ids):

        self.success_distances[fly_ids] = self.entry_distances[fly_ids]

    def compute_plume_distance(self,locations):
        if np.shape(self.source_pos)[1]>1: #Case where there is more than one plume

            x,y = utility.shift_and_rotate(
                    np.vstack((x,y)).T[:,:,np.newaxis],
                    self.source_pos[np.newaxis,:,:],
                    -self.wind_angle)

            #as of above line, shape of x and y is each (# flies) x (# sources)

            #For each fly, we assume it is chasing the plume whose source is closest
            # to it but where the coordinates in that plume's coordinate system is positive
            x[x<0.] = np.inf
            y[x<0.] = np.inf

            closest_plume_xs,inds = np.min(x,axis=1)
            closest_plume_ys = y[np.argmin(x,axis=1)]
