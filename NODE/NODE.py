import torch 
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint as odeint
import os
import copy

# Data
from lasa_preprocessing import LASA

os.environ.update({"KMP_DUPLICATE_LIB_OK":"TRUE"})

# ode structures
def uuv(t,y):
    if y.dim() == 1:
        y = y.reshape(1,-1)
    return torch.stack([y[:,1],-y[:,0]-y[:,1]-2*y[:,1]*torch.abs(y[:,1])], axis = 1)

class wrap_ode(torch.nn.Module):
    def __init__(self, ode):
        super(wrap_ode,self).__init__()
        self.ode = ode
    def forward(self, t, x):
        return self.ode(t,x)

class wrap_time_indep_ode(torch.nn.Module):
    def __init__(self, ode):
        super(wrap_time_indep_ode,self).__init__()
        self.ode = ode
    def forward(self, t, x):
        return self.ode(x)

def make_ode_data(ode, X0, n_points, dt):
    SOL = {key:[] for key in ["x","dxdt","t"]}
    t = torch.arange(0,n_points*dt,dt).to(X0.device)
    for x0 in X0:
        sol = odeint(ode, x0, t, method="rk4")
        SOL["x"].append(sol)
        SOL["dxdt"].append(ode(t, sol))
        SOL["t"].append(t)
    return SOL

# model structures
class model(torch.nn.Module):
    def __init__(self, A_init=None, diffeom = None):
        super().__init__()
        self.A = torch.nn.Parameter(A_init)
        self.diffeo = diffeom
    @torch.enable_grad()
    def dxdt_conjugate(self,x):
        if x.dim()==1:
            x = x.unsqueeze(0)
        elif x.dim() >2:
            x = x.reshape(-1,x.size()[-1])
        A = self.A
        y = self.diffeo(x)
        dydt = (A @ y.T).T
        J_y_x = get_batch_jacobian(self.diffeo, x, x.size()[-1])
        dxdt = torch.linalg.solve(J_y_x + 1e-8*torch.eye(J_y_x.size()[-1], device=x.device).unsqueeze(0).repeat(x.size()[0],1,1), dydt)
        return dxdt
    @torch.no_grad()
    def forward(self,x): #dxdt_conjugate_ng
        if x.dim()==1:
            x = x.unsqueeze(0)
        elif x.dim() >2:
            x = x.reshape(-1,x.size()[-1])
        A = self.A
        y = self.diffeo(x)
        dydt = (A @ y.T).T
        J_y_x = get_batch_jacobian(self.diffeo, x, x.size()[-1])
        dxdt = torch.linalg.solve(J_y_x + 1e-8*torch.eye(J_y_x.size()[-1], device=x.device).unsqueeze(0).repeat(x.size()[0],1,1), dydt)
        return dxdt

class diffeo(torch.nn.Module):
    def __init__(self, odef, Tend):
        super().__init__()
        self.odef = odef
        self.T = torch.tensor([0,Tend]).to(torch.float)
    def forward(self,x):
        return odeint(self.odef, x, self.T.to(x.device), method="rk4", options={"step_size":0.5})[1] # lets not be too accurate

@torch.enable_grad()
def get_batch_jacobian(f, x, nout):
    # from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
    # replace with a functorch: jacref / torch.vmap on functional.jacobian as soon as stable
    # currently very unstable / in only in nightly and also unstable
    B = x.size()[0]
    x = x.unsqueeze(1).repeat(1, nout, 1)
    x.requires_grad_(True)
    y = f(x)
    I_B = torch.eye(nout, device=x.device).unsqueeze(0).repeat(B,1,1)
    J = torch.autograd.grad(y,x,I_B, create_graph=True)[0]
    return J

# training structures
@torch.no_grad()
def get_data(n_traj_test, n_traj_train, n_points, dt, device):
        X0_train = torch.tensor(get_points_on_box2D_equally_spaced(0,0,3,3,n_traj_train)).to(device)
        X0_test = torch.tensor(get_points_on_box2D(0,0,3,3,n_traj_test)).to(device)
        ode = uuv

        real_ode = wrap_ode(ode)
        data_train = make_ode_data(real_ode, X0_train, n_points, dt)
        data_test = make_ode_data(real_ode, X0_test, n_points, dt)

        ### flatten to forget trajectory info
        x_train_tensor = torch.cat(data_train["x"])
        dxdt_train_tensor = torch.cat(data_train["dxdt"])
        if "u" in data_train.keys():
            u_train_tensor = torch.cat(data_train["u"])
        else:
            u_train_tensor = torch.zeros([x_train_tensor.size()[0],1])

        tensor_dataset = torch.utils.data.TensorDataset(
            x_train_tensor.to(torch.float), u_train_tensor.to(torch.float), dxdt_train_tensor.to(torch.float))
        return (X0_test, data_test, tensor_dataset)

def get_LASA_data(data_name, device, n_samples=None):

    if n_samples is None:
        Lasa = LASA(data_name)
    else:
        Lasa = LASA(data_name, n_samples)

    x_train_tensor = torch.Tensor(Lasa.x).to(device)
    xd_train_tensor = torch.Tensor(Lasa.xd).to(device)
    u_train_tensor = torch.zeros([x_train_tensor.size()[0],1]).to(device)

    tensor_dataset = torch.utils.data.TensorDataset(
        x_train_tensor, u_train_tensor, xd_train_tensor
    )

    X0_test = x_train_tensor[0,:].unsqueeze(0) # Lasa.idx[:-1]
    
    n_points = Lasa.idx[1]
    dt = Lasa.dt
    t = torch.arange(0,n_points*dt,dt).to(device)
    
    x_plot_tensor = torch.Tensor([Lasa.x[Lasa.idx[i]:Lasa.idx[i+1], :] for i in range(int(len(Lasa.idx))-1)]).to(device)
    xd_plot_tensor = torch.Tensor([Lasa.xd[Lasa.idx[i]:Lasa.idx[i+1], :] for i in range(int(len(Lasa.idx))-1)]).to(device)
    data_test = {"x": x_plot_tensor[0].unsqueeze(0), "dxdt": xd_plot_tensor[0].unsqueeze(0), "t": t} # unsqueeze(0)

    return X0_test, data_test, tensor_dataset, dt

def get_points_on_box2D_equally_spaced(centerx, centery, width, height, n):
    closure = 2*(width+height)
    closure_pos = np.linspace(0,closure - closure / n,n)
    closure_points = np.zeros((n,2))
    for i, cp in enumerate(closure_pos):
        if cp < height:
            closure_points[i] = np.array([0,cp]).reshape(1,2)
        elif cp < height+width:
            closure_points[i] = np.array([cp-height,height]).reshape(1,2)
        elif cp < height+width+height:
            closure_points[i] = np.array([width,height-(cp-height-width)]).reshape(1,2)
        else:
            closure_points[i] = np.array([width-(cp-height-width-height),0]).reshape(1,2)
    return closure_points - np.array([width/2,height/2]).reshape(1,2) + np.array([centerx,centery]).reshape(1,2)

def get_points_on_box2D(centerx, centery, width, height, n):
    a = np.concatenate([np.random.randint(1,3,(n,1))*2-3,np.random.rand(n,1)*2-1],1)
    b = np.concatenate([a[0:int(n/2),:],np.flip(a[int(n/2):n,:])])
    b = b * np.array([width,height]).reshape(1,2)/2 + np.array([centerx,centery]).reshape(1,2)
    return b

# monitoring structures
@torch.no_grad()
def log(epoch, loss, learning_model):
    gradient = torch.cat([param.grad.view(-1) for param in learning_model.parameters()])
    grad_max = torch.max(torch.abs(gradient))
    grad_norm = torch.norm(gradient,2)
    loss = loss.detach()
    print("\tEpoch {:05d} - loss: {:4e}, grad norm: {:4e}, grad max: {:4e}".format(epoch, loss, grad_norm, grad_max))

@torch.no_grad()
def test(learning_model, X0_test, data_test, n_points, dt):
    f = wrap_time_indep_ode(learning_model)
    data_pred = make_ode_data(f, X0_test.to(torch.float), n_points, dt)
    resid = torch.cat([t.flatten()-p.flatten() for t,p in zip(data_test["x"],data_pred["x"])])
    RMSE = torch.sqrt(torch.sum(resid**2 / len(resid)))
    print("\ttest RMSE: {}".format(RMSE))
    return {"data_pred": data_pred,"data_test": data_test, "RMSE": RMSE}

@torch.no_grad()
def initialize_plots(tensor_dataset, data_test, n_points, dt):
    fig, ax = plt.subplots(1,4,figsize=(3.2360679775*5,5))
    for a in ax: a.grid()
    for a, n in zip(ax, ["test x1-signals", "test x2-signals","test state space", "training state space"]): a.set_title(n)
    for i, x in enumerate(data_test["x"]):
        if i == 0:
            ax[0].plot(torch.arange(0,n_points*dt,dt),x[:,0].cpu(), "-g", label="test")
            ax[1].plot(torch.arange(0,n_points*dt,dt),x[:,1].cpu(), "-g", label="test")
            ax[2].plot(x[:,0].cpu(),x[:,1].cpu(), "-g", label="test")
        else:
            ax[0].plot(torch.arange(0,n_points*dt,dt),x[:,0].cpu(), "-g")
            ax[1].plot(torch.arange(0,n_points*dt,dt),x[:,1].cpu(), "-g")
            ax[2].plot(x[:,0].cpu(),x[:,1].cpu(), "-g")
    ax[3].scatter(tensor_dataset.tensors[0][:,0].cpu(),tensor_dataset.tensors[0][:,1].cpu(), s=0.1)
    for a in ax[0:-1]: a.legend()
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x1")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("x2")
    ax[2].set_xlabel("x1")
    ax[2].set_ylabel("x2")
    ax[3].set_xlabel("x1")
    ax[3].set_ylabel("x2")
    plt.tight_layout()    
    plt.show(block=False)
    plt.pause(0.02)
    return (fig, ax)

@torch.no_grad()
def visualize(learning_model, test_res, fig, ax, dir, epoch):
    data_pred = test_res["data_pred"]
    num_traj = len(data_pred["x"])
    for a in ax: a.relim(); a.autoscale()

    i = 0
    for x, t in zip(data_pred["x"],data_pred["t"]):
        x = x.cpu()
        t = t.cpu()
        if i == 0: # for labeling, works with one or multiple test examples
            ax[0].plot(t,x[:,0], "--r", lw=0.8, label="pred")
            ax[1].plot(t,x[:,1], "--r", lw=0.8, label="pred")
            ax[2].plot(x[:,0],x[:,1], "--r", lw=0.8, label="pred")
        else:
            ax[0].plot(t,x[:,0], "--r", lw=0.8)
            ax[1].plot(t,x[:,1], "--r", lw=0.8)
            ax[2].plot(x[:,0],x[:,1], "--r", lw=0.8)
        i += 1
    
    for a in ax[0:-1]:
        l = a.get_lines()
        if len(l) > 2*num_traj:
            l = a.get_lines()
            if len(l) > 3*num_traj:
                [traj.remove() for traj in l[num_traj:(2*num_traj)]]
                l = a.get_lines()
            for traj in l[num_traj:(2*num_traj)]:
                traj.set_color("orange") 
                traj.set_linewidth(0.5)
            l[num_traj].set_label("pred-prev")
        a.legend()

    fig.savefig(dir+str(epoch)+".png")
    plt.show(block=False)
    plt.pause(0.02)

def main():
    # parameters
    cuda_device = "0" # None for cpu, index of cuda device for gpu
    batch_size = 32
    epochs = 20000
    log_frq = 1
    test_frq = 20
    vis_frq = 20

    n_points = 235
    #n_traj_train = 10
    #n_traj_test = 1

    dir = "test_LASA_S_batch/"
    # preprocessing
    device = torch.device("cuda:"+str(cuda_device) if cuda_device is not None else "cpu")
    dir = os.path.join("results", dir)
    os.makedirs(dir, exist_ok=True)
    
    ## data collection
    #X0_test, data_test, tensor_dataset = get_data(n_traj_test, n_traj_train, n_points, dt, device)

    ## LASA data
    X0_test, data_test, tensor_dataset, dt = get_LASA_data("Sshape", device, n_points)

    fig, ax = initialize_plots(tensor_dataset, data_test, n_points, dt)
    ## model definition
    node = torch.nn.Sequential(
        torch.nn.Linear(2,50),
        torch.nn.ELU(),
        torch.nn.Linear(50,50),
        torch.nn.ELU(),
        torch.nn.Linear(50,50),
        torch.nn.ELU(),
        torch.nn.Linear(50,2),
        torch.nn.ELU(),
    )
    learned_ode = wrap_time_indep_ode(node)
    diffeom = diffeo(learned_ode, Tend=2)
    temp = torch.rand(2,2)
    A_init = - temp.T @ temp
    learning_model = model(A_init, diffeom).to(device)

    ## DL problem setup
    if batch_size is None:
        train_loader = torch.utils.data.DataLoader(
            dataset=tensor_dataset,
            batch_size=n_points,
            shuffle=True,
            drop_last=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    optimizer = torch.optim.Adam(learning_model.parameters())
    loss_fcn  = torch.nn.MSELoss()

    # training
    loss_list = [np.inf]
    for epoch in range(1,epochs+1):
        training_loss = 0
        for x_batch, u_batch, dxdt_batch in train_loader:
            x_batch, u_batch, dxdt_batch = x_batch.to(device), u_batch.to(device), dxdt_batch.to(device)
            optimizer.zero_grad()

            dxdt_pred = learning_model.dxdt_conjugate(x_batch)
            loss = loss_fcn(dxdt_batch, dxdt_pred)
            training_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        loss_list.append(training_loss)
            
        if loss_list[epoch] < min(loss_list[:-1]):
            best_model = copy.deepcopy(learning_model)
            torch.save(best_model, "best_NODE_model")

        if epoch % log_frq == 0:
            log(epoch, loss, learning_model)
        if epoch % test_frq == 0:
            test_res = test(learning_model, X0_test, data_test, n_points, dt)
            if epoch % vis_frq == 0: # should be multiple of test freq
                visualize(learning_model, test_res, fig, ax, dir, epoch)

if __name__ == "__main__":
    main()
    print("done")