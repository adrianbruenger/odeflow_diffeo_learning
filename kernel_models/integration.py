import numpy as np
import torch
from tqdm import tqdm

def explicit_euler(fcn, x_start, T, delta_t, show_progress = False):
    # computes trajectory of first order ODE in time [0, T] with timestep delta_t
    # from a given starting position x_start
    k_max = int(T/delta_t)
    if torch.is_tensor(x_start) == True:
        output_shape = list(x_start.shape) # to account for x_start being a single point
        output_shape.insert(0, k_max+1)
        y = torch.empty(output_shape).to(x_start.device)
        y[0] = x_start
        #print(x_start)
        if show_progress == True:
            for k in tqdm(range(0, k_max)):
                y[k+1] = y[k] + delta_t * fcn(y[k])
        else:
            for k in range(0, k_max):
                y[k+1] = y[k] + delta_t * fcn(y[k])
    else:
        try:
            y = np.empty([k_max+1, int(max(x_start.shape))])
            y[0] = x_start.T
            for k in range(0, k_max):
                y[k+1] = y[k] + delta_t * fcn(y[k, 0], y[k, 1])
        except:
            y = np.empty([k_max+1, int(max(x_start.shape))])
            y[0] =  x_start
            print(y)
            for k in range(0, k_max):
                y[k+1] = y[k] + delta_t * fcn(y[k])
    return y