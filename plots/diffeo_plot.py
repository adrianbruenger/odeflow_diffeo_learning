import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.gridspec import GridSpec
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_grid(ax, n, m, sub, diffeo = lambda z1, z2: np.array([z1, z2])):
    # plots grid consisting of n (morphed) horizontal- and m (morphed) vertical lines
    # sub determines the number of interpolation points between two two grid line base points
    # TODO: create axes or use given and return axes

    # euclidean frame
    x1_min = x2_min = -2
    x1_max = x2_max = 2

    # base points for grid lines on x1 and x2 axes:
    x1  = np.linspace(x1_min, x1_max, m)
    x2  = np.linspace(x2_min, x2_max, n)

    # sub base points/interpolation points on each grid line:
    nn = sub*n
    mm = sub*m
    h_line_bases = np.linspace(x1_min, x1_max, mm)
    v_line_bases = np.linspace(x2_min, x2_max, nn)

    # collections of m vertical and n horizontal lines
    v_lines = np.vstack([np.repeat(x1, nn), np.tile(v_line_bases, m)]).T.reshape(m,nn,2)
    h_lines = np.vstack([np.tile(h_line_bases, n), np.repeat(x2, mm)]).T.reshape(n,mm,2)

    # add line collections to plot
    def morph_and_add_collection(lines, c):
        morphed_lines = lines.copy()
        for i, line in enumerate(lines):
            for j in range(0, line.shape[0]):
                morphed_lines[i,j,:] = diffeo(line[j,0], line[j,1])
        lc = mc.LineCollection(morphed_lines, colors = c, lw = .5)
        ax.add_collection(lc)
        return morphed_lines

    m_v_lines = morph_and_add_collection(v_lines, c = "red")
    m_h_lines = morph_and_add_collection(h_lines, c = "black")

    # morphed frame
    m_x1_min = min(m_h_lines[:,:,0].min(), m_v_lines[:,:,0].min())
    m_x1_max = max(m_h_lines[:,:,0].max(), m_v_lines[:,:,0].max())
    m_x2_min = min(m_h_lines[:,:,1].min(), m_v_lines[:,:,1].min())
    m_x2_max = max(m_h_lines[:,:,1].max(), m_v_lines[:,:,1].max())

    ax.axes.set_aspect('equal')
    #plt.xlim(m_x1_min-.1, m_x1_max+.1)
    #plt.ylim(m_x2_min-.1, m_x2_max+.1)
    #max_val = max(np.abs([m_x1_min, m_x1_max, m_x2_min, m_x2_max])) # TODO
    #plt.xlim(-2, 1.5)
    #plt.ylim(-2, 1.5)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    return None

def explicit_euler(fcn, x_start, T, delta_t):
    # computes trajectory of first order ODE in time [0, T] with timestep delta_t
    # from a given starting position x_start
    k_max = int(T/delta_t)
    if torch.is_tensor(x_start) == True:
        y = torch.tensor(np.empty([int(max(x_start.shape)), k_max+1]).astype(np.float32))
        y[:, 0] = x_start
        #print(x_start)
        for k in range(0, k_max):
            y[:, k+1] = y[:, k] + delta_t * fcn(y[:, k])
    else:
        try:
            y = np.empty([int(max(x_start.shape)), k_max+1])
            y[:, 0] = x_start.T
            for k in range(0, k_max):
                y[:, k+1] = y[:, k] + delta_t * fcn(y[0, k], y[1, k])
        except:
            y = np.empty([int(max(x_start.shape)), k_max+1])
            y[:, 0] =  x_start
            print(y)
            for k in range(0, k_max):
                y[:, k+1] = y[:, k] + delta_t * fcn(y[:, k])
    return y

#def generate_grid(x_min, x_max, y_min, y_max, m, n = None):

def generate_traj_new(x_dot, diffeo, x_start_arr, T, delta_t, N):

    L = int(T/delta_t)
    x_traj_bundle = np.empty((N, x_start_arr.shape[0], L+1))
    x_dot_traj_bundle = np.empty((N, x_start_arr.shape[0], L+1))
    z_traj_bundle = np.empty((N, x_start_arr.shape[0], L+1))

    #rng = np.random.default_rng(seed = 42)
    for i, x_start in enumerate([x_start_arr]): #(rng.multivariate_normal(mean = x_start, cov = 0.01*np.eye(x_start.shape[0]), size = N)):
        print(f"x_start = {x_start}")
        x_traj = explicit_euler(x_dot, x_start, T, delta_t)
        print(x_traj.shape)
        x_dot_traj = x_dot(x_traj[0, :], x_traj[1, :])
        z_traj = diffeo(x_traj[0, :], x_traj[1, :])

        x_traj_bundle[i] = x_traj
        x_dot_traj_bundle[i] = x_dot_traj
        z_traj_bundle[i] = z_traj

    return x_traj_bundle, z_traj_bundle, x_dot_traj_bundle

if __name__ == "__main__":

    # example diffeo:
    lam = -1
    mu = -1
    diffeo = lambda x1, x2: np.array([x1, x2 - lam/(lam - 2*mu)*x1**2]) # z from euclidean space
    inv_diffeo = lambda z1, z2: np.array([z1, z2 + lam/(lam - 2*mu)*z1**2]) # x from Riemannian manifold
    dynamics = lambda x1, x2: np.array([mu*x1, lam*(x2 - x1**2)])
    latent_dynamics = lambda z1, z2: -1*np.array([z1, z2])

    # generate example trajectories for sanity check
    x_goal = np.array([0, 0])
    z_goal = diffeo(x_goal[0], x_goal[1])
    z_start = np.array([1.6, 1.18])
    x_start = inv_diffeo(z_start[0], z_start[1])

    L = 20
    N = 1

    # window for plotting 
    z1_min = z2_min = -2
    z1_max = z2_max = 2
    plot_res = 0.5

    z1_test = np.linspace(z1_min, z1_max, 20)
    z2_test = np.linspace(z2_min, z2_max, 20)

    fig = plt.figure(figsize=(16,9), dpi=400)
    grid = GridSpec(2, 2, fig)
    lw = 4

    # test trajectories:
    x_ex_traj_bundle, z_ex_traj_bundle, _ = generate_traj_new(dynamics, diffeo, x_start, 10, 1/(10*L), 1)

    Z1, Z2 = np.meshgrid(z1_test, z2_test)
    z = np.concatenate((Z1.flatten().reshape(-1, 1), Z2.flatten().reshape(-1, 1)), 1)
    z = diffeo(Z1, Z2)

    ax1 = fig.add_subplot(grid[0])
    #ax1.set_title(r"Learned DS on manifold: "+r"$\hat{\dot{x}} = f_{W}(x)$")
    ax1.streamplot(Z1, Z2, dynamics(Z1, Z2)[0], dynamics(Z1, Z2)[1], color="grey", linewidth=1.5, arrowsize=1.5)
    [ax1.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in x_ex_traj_bundle]
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    frame.set_aspect('equal', adjustable='box')

    ax2 = fig.add_subplot(grid[1])
    #ax2.set_title(r"stable gradient system on $\mathbb{R}^d$: "+r"$\dot{x} = -x$")
    ax2.streamplot(Z1, Z2, latent_dynamics(Z1, Z2)[0], latent_dynamics(Z1, Z2)[1], color="grey", linewidth=1.5, arrowsize=1.5)
    [ax2.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in z_ex_traj_bundle]
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    frame.set_aspect('equal', adjustable='box')

    ax3 = fig.add_subplot(grid[2])
    plot_grid(ax3, 11, 11, 20, inv_diffeo)
    [ax3.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in x_ex_traj_bundle]

    ax4 = fig.add_subplot(grid[3])
    plot_grid(ax4, 11, 11, 20)
    [ax4.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in z_ex_traj_bundle]

    plt.tight_layout()
    fig.savefig("diffeo.png")
    fig.savefig("diffeo.svg")
    #plt.show()

    fig1 = plt.figure(dpi=400)
    ax1 = fig1.add_subplot()
    #ax1.set_title(r"Learned DS on manifold: "+r"$\hat{\dot{x}} = f_{W}(x)$")
    ax1.streamplot(Z1, Z2, dynamics(Z1, Z2)[0], dynamics(Z1, Z2)[1], color="grey", linewidth=1.5, arrowsize=1.5)
    [ax1.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in x_ex_traj_bundle]
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    frame.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig1.savefig("diffeo1.png")

    fig2 = plt.figure(dpi=400)
    ax2 = fig2.add_subplot()
    #ax2.set_title(r"stable gradient system on $\mathbb{R}^d$: "+r"$\dot{x} = -x$")
    ax2.streamplot(Z1, Z2, latent_dynamics(Z1, Z2)[0], latent_dynamics(Z1, Z2)[1], color="grey", linewidth=1.5, arrowsize=1.5)
    [ax2.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in z_ex_traj_bundle]
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    frame.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig2.savefig("diffeo2.png")

    fig3 = plt.figure(dpi=400)
    ax3 = fig3.add_subplot()
    plot_grid(ax3, 11, 11, 20, inv_diffeo)
    [ax3.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in x_ex_traj_bundle]
    plt.tight_layout()
    fig3.savefig("diffeo3.png")

    fig4 = plt.figure(dpi=400)
    ax4 = fig4.add_subplot()
    plot_grid(ax4, 11, 11, 20)
    [ax4.plot(morph_traj[0, :], morph_traj[1, :], lw=lw) for morph_traj in z_ex_traj_bundle]
    plt.tight_layout()
    fig4.savefig("diffeo4.png")
