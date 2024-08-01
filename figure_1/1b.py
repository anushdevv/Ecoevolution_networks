import numpy as np
import amp_and_acc as amp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
from scipy.interpolate import interp1d
import scipy.stats as sp
import os
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline
import scipy
import est_approx as est

experiment_id = "param_graphs/8"
splits = 10
n_graphs = 800

num_vals = 1
val_sel = 0

def pfix(f, alpha, N):
    phi = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N))
        phi += tmp

    return 1 / (1 + phi)

results_path = "/Users/anushd/Documents/carja/ecoevo/archive/experiments/0/" + experiment_id + "/results/"

data = np.zeros([n_graphs, num_vals, 4])
for idx in range(splits * n_graphs):
    if os.path.exists(results_path + str(idx) + ".txt"):
        if num_vals == 1:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        else:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        data[idx % n_graphs, :, :] = np.add(data[idx % n_graphs, :, :], sim[-4:])

pf = data[:, :, 1] / (data[:, :, 0] + data[:, :, 1])
tf = (data[:, :, 2] + data[:, :, 3]) / (data[:, :, 0] + data[:, :, 1])

amps = []
for G_idx in range(n_graphs):
    G_path = "/Users/anushd/Documents/carja/ecoevo/archive/graphs/param_graphs/" + str(G_idx) + ".txt"
    G = nx.read_edgelist(G_path, nodetype=int)
    amps.append(amp.amplification_and_acceleration(G)[0])

#nets = [0, 99, 199, 299, 399, 449, 459, 469, 479, 489, 499, 509, 599, 699, 749, 799]
nets = [0, 99, 399, 449, 509, 599, 699, 799]
nets = [net + 1 for net in nets]
m_list = ["P", "D", "s", "p", "o", "^", "*", "X"]
for i in range(len(nets)-1):
    plt.scatter(amps[nets[i]:nets[i+1]], pf[nets[i]:nets[i+1], val_sel], marker=m_list[i])
plt.legend(["Erdos-Renyi",
            "Preferential attachment",
            "Bipartite",
            "Small-world",
            "Detour",
            "Star",
            "Random geometric"])


a = 0.53
s = -0.001
alphas = np.linspace(0, 2, 100)
plt.plot(alphas, 0.01 + alphas * (s/2 + (2 * a - 1) ** 2 / 3))

wm_p = pfix(1+s, a, 100)
plt.plot([0, 2], [wm_p]*2)

a = 0.5
s = -0.001
alphas = np.linspace(0, 2, 100)
#plt.plot(alphas, 0.01 + alphas * (s/2 + (2 * a - 1) ** 2 / 3))

wm_p = pfix(1+s, a, 100)
#plt.plot([0, 2], [wm_p]*2)

plt.show()
