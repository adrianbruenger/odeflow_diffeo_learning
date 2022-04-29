from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import torch
from kernel_model_automized import *

from datetime import datetime
import os


def generate_traj_bundle_no_loop(x_dot, x_start_arr, T, delta_t):

    t = torch.arange(0, T, delta_t).to(x_start_arr.device)
    x_traj_bundle = torch.transpose(odeint(x_dot, x_start_arr, t, method="rk4"), 0, 1).permute(2, 0, 1)

    return x_traj_bundle

def compute_RMSE(X, X_pred):
    RMSE = 0
    for traj, traj_pred in zip(X, X_pred):
        RMSE += 1/traj.shape[1] * sum(np.linalg.norm(traj - traj_pred, axis=0))
    return 1/X.shape[0]*RMSE

def test(model, X0_test, data_test, n_points, dt):
    wrapped_model = T_VelocityField(model)
    data_pred = generate_traj_bundle_no_loop(wrapped_model, X0_test, data_test["x"][0].shape[1]*dt, dt)
    RMSE = compute_RMSE(data_test["x"], data_pred.cpu().detach().numpy())
    return {"data_pred": data_pred,"data_test": data_test, "RMSE": RMSE}

def train_LASA(dev):
    names = ["DoubleBendedLine", "heee", "Leaf_1", "Sharpc", "Sshape"]
    for name in names:
        X0_test, data_test, _, dt, x_goal = get_LASA_data(name, device=dev)

        main(name)
        
        # trained model
        rff=False
        if rff:
            alpha = torch.load(f"alpha_{name}").to(dev)
            beta = torch.load(f"beta_{name}").to(dev)
        else:
            alpha = None
            beta = None
        
        state_dict = torch.load(f"state_dict_{name}", map_location=dev)
    
        m = int(np.sqrt(max(state_dict["V.V.weights"].shape)))
    
        trained_model = KernelModel(m, 2, l=60, rff=rff, alpha=alpha, beta=beta, normalize=True, origin=x_goal, int_method="rk4", velocity_scaling=False, device=dev)
    
        trained_model.load_state_dict(state_dict)
        trained_model.eval()

        test_res = test(trained_model, X0_test, data_test, data_test["x"].shape[1], dt) #[0,:].unsqueeze(0)
        torch.save(test_res, f"test_res_{name}")
        
def plot_LASA():
    names = ["DoubleBendedLine", "heee", "Leaf_1", "Sharpc", "Sshape"]

    fig = plt.figure(dpi=400)
    grid = GridSpec(1, 5, fig)
    
    RMSE = []
    
    for i, name in enumerate(names):
        test_res = torch.load(f"test_res_{name}")
        data_pred = test_res["data_pred"].cpu().detach().numpy()
        data_test = test_res["data_test"]
        
        RMSE.append(test_res["RMSE"])
        
        ax = fig.add_subplot(grid[i])
        [ax.plot(x[0,:], x[1,:], color="tab:blue", lw=0.6) for x in data_test["x"]]
        [ax.plot(x[0,:],x[1,:], "--r", lw=0.5) for x in data_pred]
        ax.set_xticks([]); ax.set_yticks([])
        #ax.set_title(name)
        ax.set_aspect('equal', 'box')
        
    print(np.mean(RMSE))
    print(np.var(RMSE))
    
    fig.tight_layout()
    fig.savefig("5LASA_NODE.png")
    fig.savefig("5LASA_NODE.svg")
    plt.show()


if __name__ == "__main__":
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    # directory for training results and plot
    dir = "LASA_test_files/"
    dir = os.path.join(dir, dt_string)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    #train_LASA("cpu")
    plot_LASA()
    print("Done!")