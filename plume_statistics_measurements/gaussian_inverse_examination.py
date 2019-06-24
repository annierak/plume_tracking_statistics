import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

#puff concentration 1D cross section visualization w/varying r_sq, detection_threshold

detection_threshold = 0.05
r_sq = 1.

puff_mol_amt =1.
# threshold_radii = np.sqrt(-1*np.log(detection_threshold/(big_K))*2*r_sqs)

def compute_Gaussian(r_sq,x):
    px = 0
    big_K = (puff_mol_amt / (8 * np.pi**3)**0.5) / r_sq**1.5
    return big_K*np.exp(-((x - px)**2) / (2 * r_sq))


fig,ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

inputs = np.linspace(-10,10,1000)
outputs = compute_Gaussian(r_sq,inputs)

l, = plt.plot(inputs,outputs)
thres, = plt.plot(inputs,detection_threshold*np.ones_like(inputs))

big_K = (puff_mol_amt / (8 * np.pi**3)**0.5) / r_sq**1.5
threshold_radius = np.sqrt(-1*np.log(detection_threshold/(big_K))*2*r_sq)

thres_rad = plt.axvline(threshold_radius)

threshold_ax = plt.axes([0.25, 0.1, 0.65, 0.03])#], facecolor=axcolor)
r_sq_ax = plt.axes([0.25, 0.05, 0.65, 0.03])#, facecolor=axcolor)

threshold_slider = Slider(threshold_ax, 'Detection Threshold',
    0.01, .25, valinit=0.05)
r_sq_slider = Slider(r_sq_ax, 'r^2', 0.1, 8., valinit=1)


def update(val):
    r_sq = r_sq_slider.val
    detection_threshold = threshold_slider.val
    outputs = compute_Gaussian(r_sq,inputs)
    l.set_ydata(outputs)
    thres.set_ydata(detection_threshold*np.ones_like(inputs))
    big_K = (puff_mol_amt / (8 * np.pi**3)**0.5) / r_sq**1.5
    threshold_radius = np.sqrt(-1*np.log(detection_threshold/(big_K))*2*r_sq)
    thres_rad.set_xdata(threshold_radius)

    fig.canvas.draw_idle()


threshold_slider.on_changed(update)
r_sq_slider.on_changed(update)

plt.figure()
r_sq_inputs = np.linspace(0.01,8.,1000)
big_K = (puff_mol_amt / (8 * np.pi**3)**0.5) / r_sq_inputs**1.5
threshold_radii = np.sqrt(-1*np.log(detection_threshold/(big_K))*2*r_sq_inputs)
plt.plot(r_sq_inputs,threshold_radii,'o')
plt.xlabel('Threshold Radius')
plt.ylabel('Counts')

plt.show()
