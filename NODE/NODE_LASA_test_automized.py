from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import torch
from NODE_automized import *

from datetime import datetime
import os

def train_LASA():
    names = ["DoubleBendedLine", "heee", "Leaf_1", "Sharpc", "Sshape"]
    for name in names:
        X0_test, data_test, _, dt = get_LASA_data(name, device="cuda:0")

        main(name)

        best_model = torch.load(f"best_NODE_model_{name}")

        test_res = test(best_model, X0_test, data_test, data_test["x"].shape[1], dt) #[0,:].unsqueeze(0)
        torch.save(test_res, f"test_res_{name}")
        torch.cuda.empty_cache()

def plot_LASA():
    names = ["DoubleBendedLine", "heee", "Leaf_1", "Sharpc", "Sshape"]

    fig = plt.figure(dpi=400)
    grid = GridSpec(1, 5, fig)
    
    for i, name in enumerate(names):
        
        test_res = torch.load(f"test_res_{name}", map_location = "cpu")
        data_pred = test_res["data_pred"]
        data_test = test_res["data_test"]
        
        ax = fig.add_subplot(grid[i])
        [ax.plot(x[:,0], x[:,1], color="tab:blue", lw=0.6) for x in data_test["x"]]
        [ax.plot(x[:,0],x[:,1], "--r", lw=0.5) for x in data_pred["x"]]
        ax.set_xticks([]); ax.set_yticks([])
        #ax.set_title(name)
        ax.set_aspect('equal', 'box')
    
    fig.tight_layout()
    fig.savefig("5LASA_NODE.png")
    fig.savefig("5LASA_NODE.svg")
    plt.show()


if __name__ == "__main__":
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    #dt_string = "07_04_2022__23_38_51"
    # directory for training results and plot
    dir = "LASA_test_files/"
    dir = os.path.join(dir, dt_string)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    #train_LASA()
    plot_LASA()
    print("Done!")