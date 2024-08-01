import numpy as np
import amp_and_acc as amp
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import os
import est_approx as est
import pfix_analytic as p

matplotlib.use('macosx')


def pfix(f, alpha, N):
    phi = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N))
        phi += tmp

    return 1 / (1 + phi)

##############################################

experiment_id = "results5"
splits = 1000
n_graphs_d = 30

num_vals = 2

results_path = "/Users/anushd/Documents/carja/ecoevo/ductal/" + experiment_id + "/results/"

data = np.zeros([n_graphs_d, num_vals, 4])
for idx in range(splits * n_graphs_d):
    if os.path.exists(results_path + str(idx) + ".txt"):
        if num_vals == 1:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        else:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        data[idx % n_graphs_d, :, :] = np.add(data[idx % n_graphs_d, :, :], sim[:, -4:])

pf = data[:, :, 1] / (data[:, :, 0] + data[:, :, 1])
tf = (data[:, :, 2] + data[:, :, 3]) / (data[:, :, 0] + data[:, :, 1])

##############################################

experiment_id = "results6"
splits = 100
n_graphs = 800

num_vals = 1

results_path = "/Users/anushd/Documents/carja/ecoevo/ductal/" + experiment_id + "/results/"

data = np.zeros([n_graphs, num_vals, 4])
for idx in range(splits * n_graphs):
    if os.path.exists(results_path + str(idx) + ".txt"):
        if num_vals == 1:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        else:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        data[idx % n_graphs, :, :] = np.add(data[idx % n_graphs, :, :], sim[-4:])

pf3 = data[:, :, 1] / (data[:, :, 0] + data[:, :, 1])
tf3 = (data[:, :, 2] + data[:, :, 3]) / (data[:, :, 0] + data[:, :, 1])

##############################################

experiment_id = "param_graphs/8"
splits = 10

num_vals = 1
val_sel = 0

results_path = "/Users/anushd/Documents/carja/ecoevo/archive/experiments/0/" + experiment_id + "/results/"

data = np.zeros([n_graphs, num_vals, 4])
for idx in range(splits * n_graphs):
    if os.path.exists(results_path + str(idx) + ".txt"):
        if num_vals == 1:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        else:
            sim = np.genfromtxt(results_path + str(idx) + ".txt")
        data[idx % n_graphs, :, :] = np.add(data[idx % n_graphs, :, :], sim[-4:])

pf2 = data[:, :, 1] / (data[:, :, 0] + data[:, :, 1])
tf2 = (data[:, :, 2] + data[:, :, 3]) / (data[:, :, 0] + data[:, :, 1])

##############################################

amps_d = []
for G_idx in range(n_graphs_d):
    G_path = "/Users/anushd/Documents/carja/ecoevo/ductal/graphs/" + str(G_idx) + ".txt"
    G = nx.read_edgelist(G_path, nodetype=int)
    amps_d.append(amp.amplification_and_acceleration(G)[0])
amps_d = np.array(amps_d)

amps_pa = []
for i in range(1, 11):
    G = p.pa_star(100 - i, i)
    amps_pa.append(amp.amplification_and_acceleration(G)[0])
amps_pa = np.array(amps_pa)

amps = []
for G_idx in range(n_graphs):
    G_path = "/Users/anushd/Documents/carja/ecoevo/archive/graphs/param_graphs/" + str(G_idx) + ".txt"
    G = nx.read_edgelist(G_path, nodetype=int)
    amps.append(amp.amplification_and_acceleration(G)[0])

amp_list = np.linspace(1, 2, 10)

##############################################
plt.figure()
plt.scatter(amps_d, pf[:, 0])
pf_approx = 1/100 + amp_list * (-0.001 / 2 + (2 * 0.505 - 1) ** 2 / 3)
plt.plot(amp_list, pf_approx, c="k")
plt.plot([1, 2], [pfix(1 - 0.001, 0.505, 100)] * 2)

plt.scatter(amps[399:449], pf3[399:449, 0])
plt.scatter(amps[599:699], pf3[599:699, 0])

plt.xlim([1, 2])
##############################################

plt.figure()
plt.scatter(amps_d, pf[:, 1])
plt.plot(amp_list, 1/100 + amp_list * (-0.001 / 2 + (2 * 0.53 - 1) ** 2 / 3), c="k")
plt.plot([1, 2], [pfix(1 - 0.001, 0.53, 100)] * 2)

plt.scatter(amps[399:449], pf2[399:449, 0])
plt.scatter(amps[599:699], pf2[599:699, 0])

plt.xlim([1, 2])

##############################################

"""
p_est_list = []
p_fix_list = []
for i in range(1, 11):
    print(i)
    a, b = p.pfix_full_pa_star(100 - i, i, 1 - 0.001, 0.9999)
    p_fix_list.append(a)
    p_est_list.append(b)
"""

plt.figure()
plt.xlim([1.1, 1.7])
plt.ylim([0.458, 0.473])
p_fix_list = [0.450110953915312, 0.451264086558548, 0.452416923225627, 0.45349544316268, 0.454490622022309, 0.455407877346328, 0.456255418560507, 0.457041253060055, 0.457772457502466, 0.45845509681474]
#plt.scatter(amps_pa, p_fix_list)
plt.scatter(amps_d, 1 / 2 + 1/4 * amps_d * 100 * -0.001)

#plt.plot([1.5, 2], 1 / 2 + 1/4 * np.array([1.5, 2]) * 100 * -0.001)
plt.plot([1, 2], [pfix(1 - 0.001, 0.9999, 100)] * 2)

##############################################
plt.show()
