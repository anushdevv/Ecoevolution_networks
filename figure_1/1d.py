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

#18 and 19 are final

experiment_id = "param_graphs/18"
splits = 10
n_graphs = 800

num_vals = 2

alpha_list = [0.5, 0.525]
s = -0.002

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
        data[idx % n_graphs, :, :] = np.add(data[idx % n_graphs, :, :], sim[:, -4:])

pf = data[:, :, 1] / (data[:, :, 0] + data[:, :, 1])
tf = (data[:, :, 2] + data[:, :, 3]) / (data[:, :, 0] + data[:, :, 1])

amps = []
for G_idx in range(n_graphs):
    G_path = "/Users/anushd/Documents/carja/ecoevo/archive/graphs/param_graphs/" + str(G_idx) + ".txt"
    G = nx.read_edgelist(G_path, nodetype=int)
    amps.append(amp.amplification_and_acceleration(G)[0])

#nets = [0, 99, 199, 299, 399, 449, 459, 469, 479, 489, 499, 509, 599, 699, 749, 799]

nets = [509, 699]
#nets = [399, 449, 509, 599]
nets = [net + 1 for net in nets]
for i in range(len(nets)-1):
    nets_idx = np.array(list(range(nets[i], nets[i+1])))
    print(nets_idx)
    plt.scatter(np.array(amps)[nets_idx], pf[nets_idx, 0] / pfix(1 + s, alpha_list[0], 100), c="r")
    plt.scatter(np.array(amps)[nets_idx], pf[nets_idx, 1] / pfix(1 + s, alpha_list[1], 100), c="g")
    #plt.scatter(amps[nets[i]:nets[i+1]], pf[nets[i]:nets[i+1], 2] / pfix(1 + s, alpha_list[2], 100), c="b")


amp_space = np.linspace(0, 2, 100)
plt.plot(amp_space, (1/100 + 1/2*amp_space*(s + 2/3*(2*alpha_list[0]-1)**2)) / (1/100 + 1/2*(s + 2/3*(2*alpha_list[0]-1)**2)), c="k")
plt.plot(amp_space, (1/100 + 1/2*amp_space*(s + 2/3*(2*alpha_list[1]-1)**2)) / (1/100 + 1/2*(s + 2/3*(2*alpha_list[1]-1)**2)), c="k")
#plt.plot(amp_space, (1/100 + 1/2*amp_space*(s + 2/3*(2*alpha_list[2]-1)**2)) / (1/100 + 1/2*(s + 2/3*(2*alpha_list[2]-1)**2)), c="k")

"""
plt.plot(amp_space, (1 + 100*1/2*(amp_space-1)*(0.001 + 2/3*(2*alpha_list[0]-1)**2)), c="r")
plt.plot(amp_space, (1 + 100*1/2*(amp_space-1)*(0.001 + 2/3*(2*alpha_list[1]-1)**2)), c="r")
plt.plot(amp_space, (1 + 100*1/2*(amp_space-1)*(0.001 + 2/3*(2*alpha_list[2]-1)**2)), c="r")
"""

"""
a = 0.53
s = -0.001
alphas = np.linspace(0, 2, 100)
#plt.plot(alphas, 0.01 + alphas * (s/2 + (2 * a - 1) ** 2 / 3))

wm_p = pfix(1+s, a, 100)
#plt.plot([0, 2], [wm_p]*2)

a = 0.5
s = -0.001
alphas = np.linspace(0, 2, 100)
#plt.plot(alphas, 0.01 + alphas * (s/2 + (2 * a - 1) ** 2 / 3))

wm_p = pfix(1+s, a, 100)
#plt.plot([0, 2], [wm_p]*2)
"""

plt.show()
