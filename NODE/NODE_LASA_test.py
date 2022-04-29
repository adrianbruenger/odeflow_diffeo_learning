import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import pyLasaDataset as lasa
from lasa_preprocessing import LASA


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


def generate_traj_bundle(x_dot, x_start_arr, T, delta_t):

    L = int(T/delta_t)
    x_traj_bundle = np.empty((x_start_arr.shape[0], L+1, x_start_arr[0].shape[0]))

    print(x_traj_bundle.shape)

    for i, x_start in enumerate(x_start_arr):

        #print(f"x_start = {x_start}")
        x_traj = explicit_euler(x_dot, x_start, T, delta_t)
        #print(x_traj.shape)

        if torch.is_tensor(x_traj) == True:
            x_traj_bundle[i] = x_traj.detach().numpy()
        else:
            x_traj_bundle[i] = x_traj

    return x_traj_bundle

def generate_traj_bundle_no_loop(x_dot, x_start_arr, T, delta_t):

    x_traj_bundle = explicit_euler(x_dot, x_start_arr, T, delta_t, show_progress=True)

    if torch.is_tensor(x_traj_bundle) == True:
        x_traj_bundle = x_traj_bundle#.detach().numpy()
    else:
        pass

    return x_traj_bundle


if __name__ == "__main__":
    
    dev = "cuda:0" # Test on cuda!!
    
    ############################################################################################## OLD DATA:
    # prepare NShape and Sshape data # TODO: select n points (every max_lenght/n th point)
    #demos = lasa.DataSet.__getattr__(name = "Sshape").demos
    # trajectories
    #X_start = torch.cat([torch.from_numpy(demo.pos.astype(np.float32))[:, 0].unsqueeze(1) for demo in demos], dim=1)#.reshape(1, 2, -1) # TODO might be incorrect due to reshape
    
    # velocities
    #X_dot = torch.cat([torch.from_numpy(demo.vel.astype(np.float32)) for demo in demos], dim=1).reshape(1, 2, -1)
    
    ############################################################################################## NEW DATA:
    Lasa = LASA("Sshape")
    X_start = torch.cat([torch.from_numpy(Lasa.x[idx, :]).unsqueeze(1) for idx in Lasa.idx[:-1]], dim = 1).to(dev)
    x_goal = torch.Tensor(Lasa.goal.T).to(dev)
    
    # trained model 
    trained_model = torch.load("best_NODE_model")
    
    #trained_model.V.x_train_min = torch.load("x_train_min").to("cpu")
    #trained_model.V.x_train_max = torch.load("x_train_max").to("cpu")
    
    #state_dict = torch.load("state_dict_complete", map_location=dev)
    #print(state_dict.keys())
    #print(state_dict["V.V.weights"].shape)
    #trained_weights = state_dict["V.V.weights"].to(dev)
    #trained_model.V.V.weights = torch.nn.Parameter(trained_weights)
    
    #trained_model.load_state_dict(state_dict)
    
    trained_model.eval()
    

    print("Generate trajectory bundle of learned dynamics")
    x_pred_traj_bundle = generate_traj_bundle_no_loop(trained_model, X_start, T = 10, delta_t = 0.05).cpu().detach().numpy()

    #print(x_pred_traj_bundle.shape)
    
    x = np.linspace(-0.5, 0.5, 20)

    y = np.linspace(-0.5, 0.5, 20)

    X, Y = np.meshgrid(x, y)
    xy = torch.from_numpy(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis = 1).astype(np.float32)).t().to(dev)
    print(xy.shape)
    xy_eval = trained_model(xy).cpu().detach().numpy()
    xy_learnedV_eval = trained_model.V.V(xy).cpu().detach().numpy()
    #print(xy_learnedV_eval)
    print("evaluated learned V above")
    #print(xy_eval.shape)
    print(X.shape)
    #print(xy_eval[0, :].reshape(X.shape).shape)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.quiver(X, Y, xy_eval[0, :].reshape(X.shape), xy_eval[1, :].reshape(Y.shape), units = "width", alpha=0.5)
    #ax.quiver(X, Y, xy_learnedV_eval[0, :].reshape(X.shape), xy_learnedV_eval[1, :].reshape(Y.shape), units = "width", alpha=0.8, color="g")
    #[ax.plot(demo.pos[0, :], demo.pos[1, :], c = "tab:blue") for demo in demos]
    [ax.plot(Lasa.x[Lasa.idx[i]:Lasa.idx[i+1], 0], Lasa.x[Lasa.idx[i]:Lasa.idx[i+1], 1], color="tab:blue")\
         for i in range(int(len(Lasa.idx))-1)]
    ax.scatter(Lasa.goal[0, 0], Lasa.goal[0, 1], marker=(5, 1), color='black')
    [ax.plot(traj[0, :], traj[1, :], c = "tab:red") for traj in x_pred_traj_bundle.T]
    #ax.plot(xy[:, 0], xy[:, 1], marker='.', color='k', linestyle='none')
    fig.savefig('lasa_complete.png')
    plt.show()