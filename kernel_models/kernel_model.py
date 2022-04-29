# model imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

# integration
from integration import explicit_euler
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint

# data
#import pyLasaDataset as lasa
from lasa_preprocessing import LASA
from torch.utils.data import DataLoader, TensorDataset

# avoid conflict with pytorch and numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Induced_GaussianRBF(nn.Module):
    def __init__(self, num_inducing, dim, l=20, normalize=True, inducing_states=None, device='cpu'):
        super(Induced_GaussianRBF, self).__init__()

        self.dim = dim #TODO unused
        self.l = l

        if inducing_states is None and normalize == True: # TODO inducing_states == True and normalize == False -> over range of demonstrations
            x_min = -0.5; x_max = 0.5
            y_min = -0.5; y_max = 0.5

            # inducing states
            x = np.linspace(x_min, x_max, num_inducing)
            y = np.linspace(y_min, y_max, num_inducing)
            X, Y = np.meshgrid(x, y)
            inducing_states = torch.from_numpy(np.concatenate((X.reshape(-1, 1), \
                                            Y.reshape(-1, 1)), axis = 1).astype(np.float32))
        elif inducing_states is None:
            # so far: if no normalization inducing states HAVE TO be handed in manually
            raise ValueError
        

        self.s_hat = inducing_states.to(device)

    def forward(self, s):
        '''computes value in s of Gaussian Kernel centered on inducing state s_hat
        s_hat is a torch.tensor, s can be either a numpy.array or a torch.tensor
        parameter l -> "width" of Gaussian'''

        if torch.is_tensor(s) == True:
            if(s.dim() > 1):
                norm_arr = torch.norm(s - self.s_hat.unsqueeze(-1) + 1e-22, dim = 1) # nan gradient if data matches inducing states!
                k = torch.exp(-self.l * norm_arr**2)
            else:
                norm_arr = torch.norm(s.unsqueeze(-1) - self.s_hat.unsqueeze(-1) + 1e-22, dim = 1)
                k = torch.exp(-self.l * norm_arr**2)
        else:
            raise ValueError
        return k

class RandomFourierFeatures(nn.Module): # TODO make nn.module
    def __init__(self, num_features, dim, l=30, alpha=None, beta=None, device='cpu'):
        super(RandomFourierFeatures, self).__init__()

        self.m = num_features
        self.d = dim
        
        self.l = l

        print(f'Initialize model with {self.m} Random Fourier Features. '\
         f'Length scale of approximated kernel: l = {self.l}')
        
        if alpha is None and beta is None:
            self.alpha = torch.from_numpy(np.random.multivariate_normal(mean=np.zeros(self.d), \
                cov=self.l*np.eye(self.d), size=[self.m]).astype(np.float32)).to(device)
            
            print(self.alpha.shape)

            self.beta = torch.from_numpy(np.random.uniform(0, 2*np.pi, \
                size=[self.m, 1]).astype(np.float32)).to(device)

            print(self.beta.shape)
            
            torch.save(self.alpha, "alpha_complete")
            torch.save(self.beta, "beta_complete")
        
        elif alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
            
        else:
            raise ValueError

    def forward(self, z):
        if z.dim() == 1:
            return np.sqrt(2/self.m) * torch.cos(torch.matmul(self.alpha, z).unsqueeze(1) + self.beta).t()
        else:
            return np.sqrt(2/self.m) *torch.cos(torch.matmul(self.alpha, z) \
                + self.beta.expand(-1, z.shape[1]))

class VelocityField(nn.Module):
    '''velocity field / infinitesimal generator of a smooth flow psi 
    -> smooth vector field mapping a point s to the tangent vector d(psi)/dt at s
    V = W.T @ k, with W [2 x m] weight matrix and k [m x 1]'''
    
    def __init__(self, num_features, dim, l, rff, alpha, beta, inducing_states, normalize, device):
        super(VelocityField, self).__init__()

        self.m = num_features**2 # TODO rename
        self.d = dim
        
        weights = torch.tensor(np.random.uniform(
            low = -1/np.sqrt(self.m), high = 1/np.sqrt(self.m), \
            size = [self.m, self.d]).astype(np.float32), requires_grad=True, device=device)
        self.weights = torch.nn.Parameter(weights)

        if rff == True:
            self.kernel = RandomFourierFeatures(
                self.m, self.d, l, alpha, beta, device=device)

        else:
            self.kernel = Induced_GaussianRBF(
                num_features, self.d, l, normalize, inducing_states, device=device)
        
    def forward(self, s):
        '''evaluates velocity field at s'''

        # compute kernel vector in s (Gaussian kernel respective each inducing state)
        kernel_vec = self.kernel(s)

        # compute matrix vector product of weight matrix and kernel vector
        if torch.is_tensor(kernel_vec) == True:
            return torch.matmul(self.weights.t(), kernel_vec)
        else:
            numpy_weights = self.weights.detach().numpy()
            return numpy_weights.T @ kernel_vec       
        
class T_VelocityField(nn.Module):
    '''Time dependent forward evaluation of VelocityField'''
    
    def __init__(self, V):
        super(T_VelocityField, self).__init__()
        self.V = V
        
    def forward(self, t, x):
        return self.V(x)

class NoT_VelocityField(nn.Module):
    '''Time independent forward evaluation of VelocityField'''
    
    def __init__(self, V):
        super(T_VelocityField, self).__init__()
        self.V = V
        
    def forward(self, x):
        return self.V(x)

def vector_flow(V, x, delta_t, T = 1): # TODO reverse, other integrators
    '''approximates vector flow (transport equartion) using 1/delta_t explicit Euler steps'''
    
    psi = explicit_euler(V, x, T, delta_t)[-1]
    return psi

def shift_and_scale(x, x_train_min, x_train_max):
    '''shifts and scales inputs so that the training points lay in [-0.5, 0.5]x[-0.5, 0.5]'''
    
    x_shift_scale = -0.5 + torch.div(x.sub(x_train_min.t()), (x_train_max - x_train_min).t())
    #x_shift_scale = -0.5 + torch.div(x.sub(-x_train_min.t()), (x_train_max + x_train_min).t())
    return x_shift_scale
    

class KernelModel(nn.Module):
    def __init__(self, num_features, dim, l, rff=True, alpha=None, beta=None, inducing_states=None, int_method="custom_euler", normalize=True, origin=None, velocity_scaling=False, eps = 1e-12, device="cpu"):
        super(KernelModel, self).__init__()
        
        #temp = torch.rand(2,2, device=device)
        #self.A = torch.nn.Parameter(temp.t() @ temp)
        #self.A = torch.nn.Parameter(torch.eye(2, device=device))
        
        self.device = device
        self.V = VelocityField(num_features, dim, l, rff, alpha, beta, inducing_states, normalize, self.device)
        
        self.int_method = int_method
        
        if self.int_method != "custom_euler":
            self.V = T_VelocityField(self.V)
        
        else:
            self.V = NoT_VelocityField(self.V)
        
        self.normalize = normalize
        
        if origin is None:
            self.origin = torch.zeros(self.dim, 1, device=self.device)
        else:
            self.origin = origin.to(self.device)
        
        self.vel_scale = velocity_scaling
        self.log_vel_scalar = FCNN(dim, 1, 100, act='leaky_relu', device=self.device)
        self.vel_scalar = lambda x: torch.exp(self.log_vel_scalar(x)) + eps

    def forward(self, x_i): # (TODO notation: rename input_point in x/z)
        '''predicts x_dot_i for a given [dim-dimensional x #points] trajectory/
        array of points (x_ij) x_i'''
        
        #if self.normalize == True:
        #    x_i = shift_and_scale(x_i, self.x_train_min, self.x_train_max)
        
        x_i.requires_grad_(True) # for computation of jacobian using autograd.grad
        
        # diffeo at x_i
        if self.int_method == "custom_euler":
            psi = vector_flow(self.V, x_i, delta_t = 0.05)
            psi_goal = vector_flow(self.V, self.origin, delta_t = 0.05)
        else:
            psi = odeint(self.V, x_i, t=torch.Tensor([0, 2]).to(self.device), method=self.int_method, options={"step_size":0.5})[1]
            psi_goal = odeint(self.V, self.origin, t=torch.Tensor([0, 2]).to(self.device), method=self.int_method, options={"step_size":0.5})[1]
        
        # approximate Jacobian in each x_ij
        J_psi_x_i_first = torch.autograd.grad(psi, x_i, torch.tensor([1, 0], device=self.device).float()\
            .expand(x_i.shape[1], x_i.shape[0]).t(), create_graph = True)[0]
        J_psi_x_i_second = torch.autograd.grad(psi, x_i, torch.tensor([0, 1], device=self.device).float()\
            .expand(x_i.shape[1], x_i.shape[0]).t(), create_graph = True)[0]
        J_psi_x_i = torch.cat((J_psi_x_i_first, J_psi_x_i_second), dim = 0).t()
        J_psi_x_i = J_psi_x_i.reshape((x_i.shape[1], x_i.shape[0] , x_i.shape[0]))
        
        J_inv = torch.inverse(J_psi_x_i)# + 1e-12*torch.eye(J_psi_x_i.size()[-1], device=self.device))
        psi_eff = psi.sub(psi_goal)#/torch.norm(x_i, dim=0)
        
        #z_dot = self.A @ psi_eff

        x_dot_pred = (-1 * torch.matmul(J_inv, psi_eff.t().unsqueeze(-1))).squeeze(-1).t()
        #x_dot_pred = (-1 * torch.matmul(J_inv, z_dot.t().unsqueeze(-1))).squeeze(-1).t()
        
        if self.vel_scale:
            x_dot_pred = self.vel_scalar(x_i.t()).squeeze() * x_dot_pred
        

        # IF JACOBIAN ISNT INVERTIBLE (!!! NO DIFFEO !!!):
        #G_inv = torch.inverse(torch.matmul(torch.transpose(J_psi_x_i, 1, 2), J_psi_x_i))
        #x_dot_pred = (-1 * torch.matmul(torch.matmul(G_inv, torch.transpose(J_psi_x_i, 1, 2)), psi_eff.t().unsqueeze(-1))).squeeze(-1).t()

        return x_dot_pred

class FCNN(nn.Module):
    '''2-layer fully connected neural network for velocity scaling'''

    def __init__(self, in_dim, out_dim, hidden_dim, act='tanh', device='cpu'):
        super(FCNN, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
                       'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

        act_func = activations[act]
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, device=device), act_func(),
            nn.Linear(hidden_dim, hidden_dim, device=device), act_func(),
            nn.Linear(hidden_dim, out_dim, device=device)
        )

    def forward(self, x):
        return self.network(x)


def train(model, train_dataset, n_samples, opt, loss_fcn, n_epochs=10000, batch_size=None, shuffle=True, loss_clip = 1e3, clip_gradient = True, clip_value_grad = 0.1):

    #n_samples = len(train_dataset)

    if batch_size is None:
        train_loader = DataLoader(dataset=train_dataset, batch_size=n_samples, shuffle=shuffle, drop_last=False)
        mean_flag = False
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        mean_flag = True

    loss_list = []

    # progress bar setup
    progress_bar = tqdm(total=n_epochs, desc="Training progress", position=0, leave=False)

    for epoch in range(0, n_epochs):
        
        training_loss = 0

        for x_batch, x_dot_batch in train_loader: # TODO normalization in batch as in E-Flows
            
            import time
            t1 = time.time()
            
            x_dot_pred_batch = model(x_batch.t())
            t2 = time.time()

            loss = loss_fcn(x_dot_batch.t(), x_dot_pred_batch)
            training_loss += loss

            if loss > loss_clip:
                continue
            
            # backward pass
            opt.zero_grad() # clear weight gradients
            loss.backward() # compute weight gradients

            # clip gradient based on norm
            if clip_gradient:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_value_grad
                )

            # update
            opt.step() # gradient descent step
            
        if mean_flag:
            training_loss = float(batch_size)/float(n_samples)*training_loss

        loss_list.append(training_loss.item())

        # save weights if current loss is smallest:
        if epoch > 0:
            if epoch % 100 == 0:
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.semilogy(loss_list)
                fig.savefig('loss_complete.png')
                plt.close(fig)
            if loss_list[epoch] < min(loss_list[:-1]):
                torch.save(model.state_dict(), "state_dict_complete")
            else:
                pass
        else:
            # ensures that any weights are saved
            torch.save(model.state_dict(), "state_dict_complete")

        # update progress bar
        progress_bar.set_postfix_str(f"Current MSE loss: {training_loss.item():.4f}, \
                    Best MSE loss: {min(loss_list):.4f}")
        progress_bar.update()

    # save best weights (respective minimal MSE loss)
    best_weights = torch.load("state_dict_complete")
    torch.save(best_weights, "best_state_dict_complete")

    return best_weights, loss_list


if __name__ == "__main__":

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # for gpu usage

    d = 2
    m = 27 # sqrt 720
    
    x_min =-44
    x_max = 0
    y_min =-44
    y_max = 0


    # inducing states
    #x = np.linspace(x_min, x_max, m)

    #y = np.linspace(y_min, y_max, m)

    #X, Y = np.meshgrid(x, y)
    #xy = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis = 1)
    
    #fig = plt.figure()
    #ax = fig.add_subplot()
    #ax.plot(xy[:, 0], xy[:, 1], marker='.', color='k', linestyle='none')
    #plt.show()
    
    #xy = torch.from_numpy(xy.astype(np.float32))
    #print("inducing states shape:")
    #print(xy.shape)
    

    # example diffeo:
    lam = -0.3
    mu = -0.7
    diffeo = lambda z1, z2: np.array([z1, z2 - lam/(lam - 2*mu)*z1**2]) # z from euclidean space
    inv_diffeo = lambda x1, x2: np.array([x1, x2 + lam/(lam - 2*mu)*x1**2]) # x from Riemannian manifold
    x_dot = lambda z1, z2: np.array([mu*z1, lam*(z2 - z1**2)]) # velocity field on manifold

    # Data preparation

    # x_start = np.array([0.5, 0.5])
    # x_ex_traj_bundle, z_ex_traj_bundle, x_dot_ex_traj_bundle = generate_traj_new(x_dot, inv_diffeo, x_start, 10, 10/15, 1)

    # x_dot_train = x_dot_ex_traj_bundle
    # x_train = x_ex_traj_bundle

    # rng = np.random.default_rng(seed = 42)
    # idx = rng.choice(np.arange(0, xy.shape[0]), size = 20)
    # print(f"idx array = {idx}")
    # x_dot_train = x_dot(xy[idx, 0], xy[idx, 1])
    # x_dot_train = x_dot_train.reshape(1, 2, -1)
    # x_train = xy[idx ,:].reshape(1, 2, -1)

    #n = 60
    #x = np.linspace(x_min, x_max, n)
    #y = np.linspace(y_min, y_max, n)
    #X, Y = np.meshgrid(x, y)

    #x_train = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis = 1)

    #print(x_train.T)
    #x_dot_train = x_dot(x_train[:, 0], x_train[:, 1])
    #print(x_dot_train)
    #x_dot_train = x_dot_train.reshape(1, 2, -1)
    #print(x_dot_train[0])
    
    #x_train = x_train.T.reshape(1, 2, -1)
    #print(x_train[0])

    # x_dot_train = x_dot(xy[:, 0], xy[:, 1])
    # x_dot_train = x_dot_train.reshape(1, 2, -1)
    # x_train = xy.reshape(1, 2, -1)

    # convert to torch tensors
    #X = torch.from_numpy(x_train.astype(np.float32)).to(dev)
    #X_dot = torch.from_numpy(x_dot_train.astype(np.float32)).to(dev)
    
    ################################################################################################# OLD DATA:
    # prepare NShape and Sshape data # TODO: select n points (every max_lenght/n th point)
    #demos = lasa.DataSet.__getattr__(name = "NShape").demos
    # trajectories
    #X = torch.cat([torch.from_numpy(demo.pos[:, ::5].astype(np.float32)) for demo in demos[:1]], dim=1).reshape(1, 2, -1).to(dev) # TODO might be incorrect due to reshape
    # velocities
    #X_dot = torch.cat([torch.from_numpy(demo.vel[:, ::5].astype(np.float32)) for demo in demos[:1]], dim=1).reshape(1, 2, -1).to(dev)
    ################################################################################################# NEW DATA:
    
    Lasa = LASA("heee")
    print(Lasa.x.shape)
    X = torch.Tensor(Lasa.x.T).to(dev)
    X_dot = torch.Tensor(Lasa.xd.T).to(dev)
    x_goal = torch.Tensor(Lasa.goal.T).to(dev)
    
    n_samples = X.shape[1]

    X_dataset = TensorDataset(X.t(), X_dot.t())
    
    model = KernelModel(m, d, l=60, rff=False, normalize=True, origin=x_goal, int_method="rk4", velocity_scaling=False, device=dev)
    '''num_features, dim, rff=True, alpha=None, beta=None, inducing_states=None, int_method="custom_euler", normalize=True, origin=None, device="cpu"'''

    # Loss and Optimizer
    
    #criterion = nn.MSELoss()#reduction="sum")
    #criterion = nn.SmoothL1Loss()
    
    custom_loss_fcn = lambda X, X_hat: 1/X.shape[1]*torch.sum(torch.norm(X - X_hat, dim = 0)**2)
    
    learning_rate = 0.0001
    weight_regularizer = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)#, weight_decay = weight_regularizer)
    
    # Training

    trained_weights, loss_list = train(model, X_dataset, n_samples, optimizer, custom_loss_fcn, n_epochs=4000)#, batch_size=32)

    print("Minimal loss:")
    print(min(loss_list))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(loss_list)
    fig.savefig('loss.png')
    plt.show()