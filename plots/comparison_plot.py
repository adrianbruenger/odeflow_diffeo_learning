import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import PercentFormatter

rc('font', family='serif', serif='cm10')
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

names = ["NODE", "RFF", "IGK"]

RMSE = [0.100768365, 0.10357413317543937, 0.13614305975121208]
var = [0.0013241137, 0.0017504164104422356, 0.0065746743654247946]

train_time_gpu = 100*np.array([73.179/51.36, 51.36/51.36, 68.37/51.36])
forward_time_gpu = 100*np.array([32.81/51.36, 27.81/51.36, 33.09/51.36])

train_time_cpu = [451.65, 1002.75, 5462.81]
forward_time_cpu = [184.69, 368.65, 3581]

fig = plt.figure(dpi = 400)
grid = GridSpec(1, 2, fig)
ax1 = fig.add_subplot(grid[0])
ax1.bar(names, RMSE, color="tab:blue")
ax1.errorbar(names, RMSE, yerr=var, fmt="", color="r")
ax1.set_title("RMSE per shape")

ax2 = fig.add_subplot(grid[1])
ax2.bar(names, train_time_gpu, label="Training epoch", color="tab:gray")
ax2.bar(names, forward_time_gpu, label="Prediction", color="tab:orange")
ax2.set_title("Computation time respective RFF"+"\n"+"(NVIDIA TITAN V)")
ax2.legend(bbox_to_anchor=(1, 1))
#formatter2 = EngFormatter(unit='/%')
ax2.yaxis.set_major_formatter(PercentFormatter())
#ax2.yaxis.set_major_formatter(formatter2)

# ax3 = fig.add_subplot(grid[2])
# ax3.bar(names, train_time_cpu, label="Training")
# ax3.bar(names, forward_time_cpu, label="Prediction")
# ax3.set_title("Computation time [ms]"+"\n"+"(intel CORE i7, 8th Gen)")

plt.tight_layout()

fig.savefig("comparison.png")
plt.show()
