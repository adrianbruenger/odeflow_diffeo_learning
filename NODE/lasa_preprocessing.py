import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pyLasaDataset as lasa
import torch


class LASA:
    '''
    Loads LASA dataset
    NOTE: The data has been smoothed and normalized to stay within [-0.5, 0.5]

    from https://github.com/mrana6/euclideanizing_flows/blob/master/euclideanizing_flows
    '''


    def __init__(self, data_name, n_samples=235):
        [self.x, self.xd, self.idx, self.dt, self.goal, self.scaling, self.translation] = \
            self.load_data(data_name, n_samples)

        self.n_dims = self.x.shape[1]
        self.n_pts = self.x.shape[0]

    def plot_data(self):
        fig = plt.figure()
        ax = fig.gca()
        x = (self.x - self.translation) / self.scaling

        ax.scatter(x[:, 0], x[:, 1], color='r', s=0.5)

    def load_data(self, data_name, n_samples):
        downsample_rate = 4
        start_cut = 5
        end_cut = 10
        tol_cutting = 1.

        dataset = lasa.DataSet.__getattr__(name = data_name).demos

        num_demos = int(len(dataset))
        d = dataset[0].pos.shape[0]

        #for i in range(0, num_demos):
        #    print(dataset[i].t[0, 1] - dataset[0].t[0, 0])
        #    # not the same?!

        dt_old = dataset[0].t[0, 1] - dataset[0].t[0, 0]
        dt = round(dt_old * downsample_rate, 2)

        x = np.empty((d, 0))
        idx = [0]

        for i in range(num_demos):
            demo = dataset[i].pos[:, ::downsample_rate]
            num_pts = demo.shape[1]

            demo_smooth = np.zeros_like(demo)
            window_size = int(2 * (25. * num_pts / 150 // 2) + 1)  # just an arbitrary heuristic (can be changed)
            for j in range(d):
                demo_smooth[j, :] = savgol_filter(demo[j, :], window_size, 3)

            demo_smooth = demo_smooth[:, start_cut:-end_cut]

            # upsampling
            t_end = demo_smooth.shape[1]*dt
            t = np.arange(0, t_end, dt)

            demo_smooth_fcn_x_t = interp1d(t, demo_smooth[0,:], kind="cubic")
            demo_smooth_fcn_y_t = interp1d(t, demo_smooth[1,:], kind="cubic")

            t_new = np.linspace(0, t[-1], n_samples)
            dt_new = t_end / n_samples

            demo_smooth_upsample_x = demo_smooth_fcn_x_t(t_new)
            demo_smooth_upsample_y = demo_smooth_fcn_y_t(t_new)

            demo_smooth_upsample = np.vstack((demo_smooth_upsample_x, demo_smooth_upsample_y))

            # plot upsampled shape
            #fig = plt.figure()
            #ax = fig.add_subplot()
            #ax.plot(demo_smooth_upsample[0,:], demo_smooth_upsample[1,:], color="tab:orange", alpha=0.9, lw=0.5)
            #ax.scatter(demo_smooth[0,:], demo_smooth[1,:], s=0.8, color="tab:green")
            #plt.show()

            demo_pos = demo_smooth_upsample
            demo_vel = np.diff(demo_smooth_upsample, axis=1) / dt_new
            demo_vel_norm = np.linalg.norm(demo_vel, axis=0)
            ind = np.where(demo_vel_norm > tol_cutting * 150. / n_samples)

            demo_pos = demo_pos[:, np.min(ind):(np.max(ind) + 2)]
            tmp = demo_pos
            for j in range(d):
                tmp[j, :] = savgol_filter(tmp[j, :], window_size, 3)
            demo_pos = tmp

            demo_pos = demo_pos - demo_pos[:, -1].reshape(-1, 1)
            x = np.concatenate((x, demo_pos), axis=1)
            idx.append(x.shape[1])

        minx = np.min(x, axis=1).reshape(-1, 1)
        maxx = np.max(x, axis=1).reshape(-1, 1)

        scaling = 1. / (maxx - minx)
        translation = -minx / (maxx - minx) - 0.5

        x = x*scaling + translation
        xd = np.empty((d, 0))

        for i in range(num_demos):
            demo = x[:, idx[i]:idx[i + 1]]
            demo_vel = np.diff(demo, axis=1) / dt_new
            demo_vel = np.concatenate((demo_vel, np.zeros((d, 1))), axis=1)

            xd = np.concatenate((xd, demo_vel), axis=1)

        x = x.T
        xd = xd.T
        goal = x[-1].reshape(1, -1)
        scaling = scaling.T
        translation = translation.T

        return [x.astype(np.float32), xd.astype(np.float32), idx, dt_new, goal, scaling, translation]

if __name__ == "__main__":

    # plot
    Lasa = LASA("Sshape", n_samples=1000)
    print(Lasa.goal)
    x = Lasa.x
    print(Lasa.x.shape[0])
    print(len(Lasa.idx)-1)
    x_dot = Lasa.xd
    x_dot_norm = np.linalg.norm(x_dot, axis=1)

    # sample inducing states along mean trajectory

    x_tensor = torch.Tensor([x[Lasa.idx[i]:Lasa.idx[i+1], :] for i in range(int(len(Lasa.idx))-1)])
    print(x_tensor.shape)
    x_mean_tensor = torch.mean(x_tensor, dim=0)
    print(x_mean_tensor.shape)

    m = 10
    m = m**2

    num_samples = max(x_mean_tensor.shape)

    if m > num_samples:
        raise ValueError
    
    idx_array = torch.floor(torch.linspace(0, num_samples-1, m)).type(torch.LongTensor)

    inducing_states = torch.normal(mean=x_mean_tensor[idx_array, :], std=0.01)
    print(inducing_states.shape)
    fig = plt.figure()
    plt.scatter(inducing_states[:, 0], inducing_states[:, 1], s = 0.5, color = 'black')
    #plt.show()
    #assert False


    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    [ax1.plot(x[Lasa.idx[i]:Lasa.idx[i+1], 0], x[Lasa.idx[i]:Lasa.idx[i+1], 1])\
         for i in range(int(len(Lasa.idx))-1)]

    ax1.plot(x_mean_tensor[:, 0], x_mean_tensor[:, 1], color='black', lw = 2)

    ax1.scatter(Lasa.goal[0, 0], Lasa.goal[0, 1], marker=(5, 1), color='black')
    ax2 = fig.add_subplot(212)
    [ax2.plot(np.arange(0, Lasa.dt*Lasa.x.shape[0]/(len(Lasa.idx)-1), Lasa.dt), x_dot_norm[Lasa.idx[i]:Lasa.idx[i+1]]) for i in range(int(len(Lasa.idx))-1)]

    #Lasa.plot_data()
    plt.show()