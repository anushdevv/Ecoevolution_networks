import networkx as nx
import numpy as np
from scipy.optimize import fsolve

def amplification_and_acceleration(G):
    dlist = dict(G.degree())
    N = len(dlist)
    corr = np.zeros((N, N))
    p = np.zeros(N)

    for d in dlist:
        p[dlist[d]] += 1

    for e in G.edges:
        d0 = dlist[e[0]]
        d1 = dlist[e[1]]
        corr[d0, d1] = corr[d0, d1] + 1 / p[d0] / d0
        corr[d1, d0] = corr[d1, d0] + 1 / p[d1] / d1

    p = p / N

    idx = np.nonzero(p)[0]
    p = p[idx]
    corr = corr[idx][:, idx]
    
    
    amp = (p.T @ corr @ (1 / idx)[:,None]) / (p.T @ corr @ (1 / idx**2)[:,None])
    amp = (amp / idx) @ p
    
    acc = (p.T @ (corr) @ (1 / idx**2)[:,None]) #* (1 / idx) @ p
    acc = (acc ) / ((p / idx).sum())**2
    
    return amp, 1/acc


def abstract_graph(G):
    phi = nx.transitivity(G)
    dlist = dict(G.degree())
    N = len(dlist)
    n_edge = np.zeros((N, N))
    n_node = np.zeros(N)

    for d in dlist:
        n_node[dlist[d]] += 1

    for e in G.edges:
        d0 = dlist[e[0]]
        d1 = dlist[e[1]]
        n_edge[d0, d1] += 1 
        n_edge[d1, d0] += 1

    idx = np.nonzero(n_node)[0]
    n_node = n_node[idx]
    n_edge = n_edge[idx][:, idx]
    return n_node, idx, n_edge