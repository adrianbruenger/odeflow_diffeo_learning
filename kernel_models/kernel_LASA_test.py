import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint as odeint
from kernel_model import KernelModel, T_VelocityField
from tqdm import tqdm

from lasa_preprocessing import LASA


def generate_traj_bundle_no_loop(x_dot, x_start_arr, T, delta_t):

    t = torch.arange(0, T, delta_t).to(x_start_arr.device)
    x_traj_bundle = torch.transpose(odeint(x_dot, x_start_arr, t, method="rk4"), 0, 1).permute(2, 0, 1)

    return x_traj_bundle

def compute_RMSE(X, X_pred):
    RMSE = 0
    for traj, traj_pred in zip(X, X_pred):
        RMSE += 1/traj.shape[1] * sum(np.linalg.norm(traj - traj_pred, axis=0))
    return 1/X.shape[0]*RMSE
        


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
    Lasa = LASA("heee")
    
    dt = Lasa.dt
    T_end = Lasa.idx[1]*Lasa.dt
    
    X_start = torch.cat([torch.from_numpy(Lasa.x[idx, :]).unsqueeze(1) for idx in Lasa.idx[:-1]], dim = 1).to(dev)
    x_goal = torch.Tensor(Lasa.goal.T).to(dev)
    
    # trained model
    alpha = torch.load("alpha_complete").to(dev)
    beta = torch.load("beta_complete").to(dev)
    
    state_dict = torch.load("state_dict_complete", map_location=dev)
    #print(state_dict["A"])
    
    m = int(np.sqrt(max(state_dict["V.V.weights"].shape)))
    print(m)
    
    trained_model = KernelModel(m, 2, l=30, rff=False, alpha=alpha, beta=beta, normalize=True, origin=x_goal, int_method="rk4", velocity_scaling=False, device=dev)
    
    trained_model.load_state_dict(state_dict)
    trained_model.eval()
    
    wrapped_trained_model = T_VelocityField(trained_model)
    
    # predicted trajectories
    print("\nGenerate trajectory bundle of learned dynamics...")
    x_pred_traj_bundle = generate_traj_bundle_no_loop(wrapped_trained_model, X_start, T = T_end, delta_t = dt).cpu().detach().numpy()
    print(x_pred_traj_bundle.shape)
    print("Done!\n")
    
    # original trajectories
    x_traj_bundle = np.transpose(np.array([Lasa.x[Lasa.idx[i]:Lasa.idx[i+1], :] for i in range(int(len(Lasa.idx))-1)]), (0, 2, 1))
    
    RMSE = compute_RMSE(x_traj_bundle, x_pred_traj_bundle)
    print(f"RMSE: {RMSE:.4f}")
    
    # plot
    x = np.linspace(-0.5, 0.5, 20)
    y = np.linspace(-0.5, 0.5, 20)
    X, Y = np.meshgrid(x, y)
    xy = torch.from_numpy(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis = 1).astype(np.float32)).t().to(dev)
    xy_eval = trained_model(xy).cpu().detach().numpy()
    xy_learnedV_eval = trained_model.V.V(xy).cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.quiver(X, Y, xy_eval[0, :].reshape(X.shape), xy_eval[1, :].reshape(Y.shape), units = "width", alpha=0.5)
    #ax.quiver(X, Y, xy_learnedV_eval[0, :].reshape(X.shape), xy_learnedV_eval[1, :].reshape(Y.shape), units = "width", alpha=0.8, color="g")
    #ax.streamplot(X, Y, xy_eval[0, :].reshape(X.shape), xy_eval[1, :].reshape(Y.shape), linewidth=0.5, color="black")
    #[ax.plot(demo.pos[0, :], demo.pos[1, :], c = "tab:blue") for demo in demos]
    [ax.plot(Lasa.x[Lasa.idx[i]:Lasa.idx[i+1], 0], Lasa.x[Lasa.idx[i]:Lasa.idx[i+1], 1], color="tab:blue")\
         for i in range(int(len(Lasa.idx))-1)]
    ax.scatter(Lasa.goal[0, 0], Lasa.goal[0, 1], marker=(5, 1), color='black')
    [ax.plot(traj[0, :], traj[1, :], c = "tab:red") for traj in x_pred_traj_bundle]
    #ax.plot(xy[:, 0], xy[:, 1], marker='.', color='k', linestyle='none')
    fig.savefig('lasa_complete.png')
    plt.show()