import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import est_approx as est
from scipy.optimize import fsolve
from math import comb
import sympy as syp
import symengine as sp

def network_diffusion_steady(G, pop_state, alpha, S, D):
    L = nx.laplacian_matrix(G).toarray()

    alpha_state = np.zeros(len(pop_state))
    for idx in range(0, len(alpha)):
        alpha_state += (pop_state == idx) * alpha[idx]

    a = np.add(D * L, np.diag(alpha_state))
    b = np.array([S] * len(pop_state))

    C = np.linalg.solve(a, b)

    return C, alpha_state


def network_fit(G, pop_state, alpha, S, D):
    f = np.zeros(len(pop_state))
    for idx, alpha_sel in enumerate(alpha):
        C, alpha_state = network_diffusion_steady(G, pop_state, alpha_sel, S[idx], D[idx])
        f = np.add(f, np.multiply(C, alpha_state))

    return f


def p_t_tc_fix(G, f, a):
    N = len(list(G))

    states = []
    for i in range(N + 1):
        sub_states = [set(j) for j in combinations(range(N), i)]
        states += sub_states

    vs = len(states)
    inv_deg = 1 / np.array(list(dict(nx.degree(G)).values()))

    M = np.zeros([vs, vs])
    for i in range(vs):
        for j in range(i + 1, vs):
            d = states[j].difference(states[i])
            d_len = len(states[j]) - len(states[i])
            if len(d) == 1 and d_len == 1:
                nn = set(G.neighbors(list(d)[0]))
                nn_int = nn.intersection(states[i])
                nn_diff = nn.difference(states[i])

                p_i = (f * est.r(a, len(states[i]) / N)) / (f * est.r(a, len(states[i]) / N) * len(states[i]) + N - len(states[i]))
                p_j = 1 / (f * est.r(a, len(states[j]) / N) * len(states[j]) + N - len(states[j]))

                if len(nn_int) == 0:
                    p = p_j * np.sum(inv_deg[list(nn_diff)])
                    M[j, i] = p
                elif len(nn_int) == len(nn):
                    p = p_i * np.sum(inv_deg[list(nn_int)])
                    M[i, j] = p
                else:
                    p1 = p_i * np.sum(inv_deg[list(nn_int)])
                    p2 = p_j * np.sum(inv_deg[list(nn_diff)])
                    M[i, j] = p1
                    M[j, i] = p2

    z = np.sum(M, 1)
    for i in range(vs):
        M[i, i] = 1 - z[i]

    Q = M[1:(vs - 1), 1:(vs - 1)]
    S = np.vstack([M[1:-1, 0], M[1:-1, -1]]).T
    tfixation = np.linalg.inv((np.eye(vs - 2) - Q))
    pfixation = tfixation @ S

    ctfixation = np.zeros(np.shape(tfixation))
    for i in range(np.shape(tfixation)[0]):
        for j in range(np.shape(tfixation)[1]):
            ctfixation[i, j] = tfixation[i, j] * pfixation[j, 1] / pfixation[i, 1]

    return np.array([np.mean(pfixation[:N, 1]), np.mean(np.sum(tfixation[:N, :], 1)), pfixation[:N, 1] @ np.sum(ctfixation[:N, :], 1) / np.sum(pfixation[:N, 1])])


def pfix_full(G, f, a, phi=None):
    N = len(list(G))

    states = []
    n_mut = []
    for i in range(N + 1):
        sub_states = [set(j) for j in combinations(range(N), i)]
        states += sub_states
        n_mut += [i] * len(sub_states)

    vs = len(states)
    inv_deg = 1 / np.array(list(dict(nx.degree(G)).values()))

    M = np.zeros([vs, vs])
    for i in range(vs):
        for j in range(i + 1, vs):
            d = states[j].difference(states[i])
            d_len = len(states[j]) - len(states[i])
            if len(d) == 1 and d_len == 1:
                nn = set(G.neighbors(list(d)[0]))
                nn_int = nn.intersection(states[i])
                nn_diff = nn.difference(states[i])

                p_i = (f * est.r(a, len(states[i]) / N)) / (f * est.r(a, len(states[i]) / N) * len(states[i]) + N - len(states[i]))
                p_j = 1 / (f * est.r(a, len(states[j]) / N) * len(states[j]) + N - len(states[j]))

                if len(nn_int) == 0:
                    p = p_j * np.sum(inv_deg[list(nn_diff)])
                    M[j, i] = p
                elif len(nn_int) == len(nn):
                    p = p_i * np.sum(inv_deg[list(nn_int)])
                    M[i, j] = p
                else:
                    p1 = p_i * np.sum(inv_deg[list(nn_int)])
                    p2 = p_j * np.sum(inv_deg[list(nn_diff)])
                    M[i, j] = p1
                    M[j, i] = p2

    z = np.sum(M, 1)
    for i in range(vs):
        M[i, i] = 1 - z[i]

    # Fixation probability
    Q = M[1:-1, 1:-1]
    S = np.vstack([M[1:-1, 0], M[1:-1, -1]]).T
    p_absorb = np.linalg.inv((np.eye(vs - 2) - Q)) @ S
    pfix_mat = np.mean(p_absorb[:N, 1])

    # Intrinsic fixation probability
    if phi == None:
        if a == 0.5:
            phi = int(N / 2) # quasi-equilibrium case
        else:
            phi = int(np.round(N * (0.5 + 0.5 * (f - 1) / (f + 1) * (2 * a - 1) ** -2)))
    int_idx = np.array(n_mut[1:-1]) == phi
    pint_mat = p_absorb[int_idx, 1]

    # Establishment probability
    int_idx = np.where(int_idx)[0]
    int_idx_min = np.min(int_idx) + 1
    Q = M[1:int_idx_min, 1:int_idx_min]
    S = np.vstack([M[1:int_idx_min, 0], M[1:int_idx_min, int_idx + 1].T])
    p_absorb = np.linalg.inv((np.eye(int_idx_min - 1) - Q)) @ S.T

    pest_mat = p_absorb[:N, 1:]

    #return np.sum(p_absorb[:N, 1:], 0) / np.sum(p_absorb[:N, 1:]), np.array(states)[int_idx + 1]
    return np.array([np.mean(pfix_mat), np.mean(np.sum(pest_mat, 1)), pint_mat @ np.sum(p_absorb[:N, 1:], 0) / np.sum(p_absorb[:N, 1:])])


def local_p_t_tc_fix(G, f, a):
    N = len(list(G))

    states = []
    for i in range(N + 1):
        sub_states = [set(j) for j in combinations(range(N), i)]
        states += sub_states

    vs = len(states)
    inv_deg = 1 / np.array(list(dict(nx.degree(G)).values()))

    M = np.zeros([vs, vs])
    for i in range(vs):
        for j in range(i + 1, vs):
            d = states[j].difference(states[i])
            d_len = len(states[j]) - len(states[i])
            if len(d) == 1 and d_len == 1:
                nn = set(G.neighbors(list(d)[0]))
                nn_int = nn.intersection(states[i])
                nn_diff = nn.difference(states[i])

                p_i = (f * est.r(a, len(states[i]) / N)) / (f * est.r(a, len(states[i]) / N) * len(states[i]) + N - len(states[i]))
                p_j = 1 / (f * est.r(a, len(states[j]) / N) * len(states[j]) + N - len(states[j]))

                if len(nn_int) == 0:
                    p = p_j * np.sum(inv_deg[list(nn_diff)])
                    M[j, i] = p
                elif len(nn_int) == len(nn):
                    p = p_i * np.sum(inv_deg[list(nn_int)])
                    M[i, j] = p
                else:
                    p1 = p_i * np.sum(inv_deg[list(nn_int)])
                    p2 = p_j * np.sum(inv_deg[list(nn_diff)])
                    M[i, j] = p1
                    M[j, i] = p2

    z = np.sum(M, 1)
    for i in range(vs):
        M[i, i] = 1 - z[i]

    Q = M[1:(vs - 1), 1:(vs - 1)]
    S = np.vstack([M[1:-1, 0], M[1:-1, -1]]).T
    tfixation = np.linalg.inv((np.eye(vs - 2) - Q))
    pfixation = tfixation @ S

    ctfixation = np.zeros(np.shape(tfixation))
    for i in range(np.shape(tfixation)[0]):
        for j in range(np.shape(tfixation)[1]):
            ctfixation[i, j] = tfixation[i, j] * pfixation[j, 1] / pfixation[i, 1]

    return np.array([np.mean(pfixation[:N, 1]), np.mean(np.sum(tfixation[:N, :], 1)), np.mean(np.sum(ctfixation[:N, :], 1))])


def step(G, f, a, start, stop):
    if start == 1:
        _, p, _ = pfix_full(G, f, a, stop)

    elif stop == len(G):
        _, _, p = pfix_full(G, f, a, start)

    else:
        fix, p1, _ = pfix_full(G, f, a, start)
        _, _, p3 = pfix_full(G, f, a, stop)
        p = fix / (p1 * p3)

    return p


def pfix_intersect(G, a):
    def delta(f):
        d = p_t_tc_fix(G, f, a)[0] - p_t_tc_fix(nx.complete_graph(len(G)), f, a)[0]
        return d
    s = fsolve(delta, [1]) - 1
    return s


def p_t_tc_fix_sp1(G, n2, f, a):
    n1 = len(list(G))
    N = n1 + n2

    states = []
    for i in range(n1 + 1):
        sub_states = [set(j) for j in combinations(range(n1), i)]
        states += sub_states

    vs = len(states)
    inv_deg = 1 / (np.array(list(dict(nx.degree(G)).values())) + n2)

    M = np.zeros([vs * (n2 + 1), vs * (n2 + 1)])
    for k in range(n2 + 1):
        sub_M = np.zeros([vs, vs])
        for i in range(vs):
            for j in range(i + 1, vs):
                d = states[j].difference(states[i])
                d_len = len(states[j]) - len(states[i])
                if len(d) == 1 and d_len == 1:
                    nn = set(G.neighbors(list(d)[0]))
                    nn_int = nn.intersection(states[i])
                    nn_diff = nn.difference(states[i])

                    len_i = len(states[i]) + k
                    len_j = len(states[j]) + k
                    p_i = (f * est.r(a, len_i / N)) / (f * est.r(a, len_i / N) * len_i + N - len_i)
                    p_j = 1 / (f * est.r(a, len_j / N) * len_j + N - len_j)

                    if len(nn_int) == 0 and k == 0:
                        p = p_j * (np.sum(inv_deg[list(nn_diff)]) + (n2 - k) / (N - 1))
                        sub_M[j, i] = p
                    elif len(nn_int) == len(nn) and k == n2:
                        p = p_i * (np.sum(inv_deg[list(nn_int)]) + k / (N - 1))
                        sub_M[i, j] = p
                    else:
                        p1 = p_i * (np.sum(inv_deg[list(nn_int)]) + k / (N - 1))
                        p2 = p_j * (np.sum(inv_deg[list(nn_diff)]) + (n2 - k) / (N - 1))
                        sub_M[i, j] = p1
                        sub_M[j, i] = p2
                    # print(sub_M)
            if k < n2:
                len_i = len(states[i]) + k

                if len(states[i]) > 0:
                    deg_norm = np.sum(inv_deg[list(states[i])])
                else:
                    deg_norm = 0

                p_i = (f * est.r(a, len_i / N)) / (f * est.r(a, len_i / N) * len_i + N - len_i)
                M[k * vs + i, (k + 1) * vs + i] = p_i * (k * (n2 - k) / (N - 1) + (n2 - k) * deg_norm)

            if k > 0:
                len_i = len(states[i]) + k

                if len(states[i]) < n1:
                    deg_norm = np.sum(inv_deg[list(set(range(n1)).difference(states[i]))])
                else:
                    deg_norm = 0

                p_j = 1 / (f * est.r(a, len_i / N) * len_i + N - len_i)
                M[k * vs + i, (k - 1) * vs + i] = p_j * ((n2 - k) * k / (N - 1) + k * deg_norm)
                #M[k * vs + i, (k - 1) * vs + i] = p_j * (N - len_i) * (k / (N - 1))

        M[k * vs: (k + 1) * vs, k * vs: (k + 1) * vs] = sub_M
        M[0, :] = 0
        M[-1, :] = 0

    z = np.sum(M, 1)
    for i in range(np.shape(M)[0]):
        M[i, i] = 1 - z[i]

    Q = M[1: -1, 1: -1]
    S = np.vstack([M[1: -1, 0], M[1: -1, -1]]).T
    tfixation = np.linalg.inv((np.eye(np.shape(Q)[0]) - Q))
    pfixation = tfixation @ S

    ctfixation = np.zeros(np.shape(tfixation))
    for i in range(np.shape(tfixation)[0]):
        for j in range(np.shape(tfixation)[1]):
            ctfixation[i, j] = tfixation[i, j] * pfixation[j, 1] / pfixation[i, 1]

    vec = np.array([1] * n1 + [n2])
    vec_norm = vec / np.sum(vec)
    idx = list(range(n1)) + [vs - 1]
    return np.array([pfixation[idx, 1] @ vec_norm, np.sum(tfixation[idx, :], 1) @ vec_norm, np.sum(ctfixation[idx, :], 1) @ vec_norm])


def p_t_tc_fix_detour(n1, n2, f, a):
    G = nx.path_graph(n1)
    G.add_edge(0, n1 - 1)
    N = n1 + n2

    states = []
    for i in range(n1 + 1):
        sub_states = [set(j) for j in combinations(range(n1), i)]
        states += sub_states

    vs = len(states)
    inv_deg = np.array([1 / 2] * n1)
    inv_deg[0] = 1 / (2 + n2)
    inv_deg[n1 - 1] = 1 / (2 + n2)
    n_mut = np.zeros(vs * (n2 + 1))

    M = np.zeros([vs * (n2 + 1), vs * (n2 + 1)])
    for k in range(n2 + 1):
        sub_M = np.zeros([vs, vs])
        for i in range(vs):

            n_mut[k * vs + i] = len(states[i]) + k

            for j in range(i + 1, vs):
                d = states[j].difference(states[i])
                d_len = len(states[j]) - len(states[i])
                if len(d) == 1 and d_len == 1:
                    nn = set(G.neighbors(list(d)[0]))
                    nn_int = nn.intersection(states[i])
                    nn_diff = nn.difference(states[i])

                    len_i = len(states[i]) + k
                    len_j = len(states[j]) + k
                    p_i = (f * est.r(a, len_i / N)) / (f * est.r(a, len_i / N) * len_i + N - len_i)
                    p_j = 1 / (f * est.r(a, len_j / N) * len_j + N - len_j)

                    if len(nn_int) == 0 and k == 0 and (d == {0} or d == {n1 - 1}):
                        p = p_j * (np.sum(inv_deg[list(nn_diff)]) + (n2 - k) / (N - 1 - (n1 - 2)))
                        sub_M[j, i] = p

                    elif len(nn_int) == len(nn) and k == n2 and (d == {0} or d == {n1 - 1}):
                        p = p_i * (np.sum(inv_deg[list(nn_int)]) + k / (N - 1 - (n1 - 2)))
                        sub_M[i, j] = p

                    elif len(nn_int) == 0 and (d != {0} and d != {n1 - 1}):
                        p = p_j * np.sum(inv_deg[list(nn_diff)])
                        sub_M[j, i] = p

                    elif len(nn_int) == len(nn) and (d != {0} and d != {n1 - 1}):
                        p = p_i * np.sum(inv_deg[list(nn_int)])
                        sub_M[i, j] = p

                    elif d == {0} or d == {n1 - 1}:
                        p1 = p_i * (np.sum(inv_deg[list(nn_int)]) + k / (N - 1 - (n1 - 2)))
                        p2 = p_j * (np.sum(inv_deg[list(nn_diff)]) + (n2 - k) / (N - 1 - (n1 - 2)))
                        sub_M[i, j] = p1
                        sub_M[j, i] = p2

                    else:
                        p1 = p_i * np.sum(inv_deg[list(nn_int)])
                        p2 = p_j * np.sum(inv_deg[list(nn_diff)])
                        sub_M[i, j] = p1
                        sub_M[j, i] = p2

            if k < n2:
                len_i = len(states[i]) + k

                deg_norm = 0
                if 0 in states[i]:
                    deg_norm += 1 / (n2 + 2)
                if n1 - 1 in states[i]:
                    deg_norm += 1 / (n2 + 2)

                p_i = (f * est.r(a, len_i / N)) / (f * est.r(a, len_i / N) * len_i + N - len_i)
                M[k * vs + i, (k + 1) * vs + i] = p_i * (k * (n2 - k) / (N - 1 - (n1 - 2)) + (n2 - k) * deg_norm)

            if k > 0:
                len_i = len(states[i]) + k

                deg_norm = 0
                if 0 not in states[i]:
                    deg_norm += 1 / (n2 + 2)
                if n1 - 1 not in states[i]:
                    deg_norm += 1 / (n2 + 2)

                p_j = 1 / (f * est.r(a, len_i / N) * len_i + N - len_i)
                M[k * vs + i, (k - 1) * vs + i] = p_j * ((n2 - k) * k / (N - 1 - (n1 - 2)) + k * deg_norm)

        M[k * vs: (k + 1) * vs, k * vs: (k + 1) * vs] = sub_M
        M[0, :] = 0
        M[-1, :] = 0

    z = np.sum(M, 1)
    for i in range(np.shape(M)[0]):
        M[i, i] = 1 - z[i]

    Q = M[1: -1, 1: -1]
    S = np.vstack([M[1: -1, 0], M[1: -1, -1]]).T
    tfixation = np.linalg.inv((np.eye(np.shape(Q)[0]) - Q))
    pfixation = tfixation @ S

    ctfixation = np.zeros(np.shape(tfixation))
    for i in range(np.shape(tfixation)[0]):
        for j in range(np.shape(tfixation)[1]):
            ctfixation[i, j] = tfixation[i, j] * pfixation[j, 1] / pfixation[i, 1]

    vec = np.array([1] * n1 + [n2])
    vec_norm = vec / np.sum(vec)
    idx = list(range(n1)) + [vs - 1]

    # Establishment probability
    int_idx1 = np.array(n_mut[1: -1]) < int(N / 2)
    int_idx1 = list(np.where(int_idx1)[0])

    int_idx2 = np.array(n_mut[1: -1]) == int(N / 2)
    int_idx2 = list(np.where(int_idx2)[0])
    int_idx2 = [0] + int_idx2
    Q2 = Q[int_idx1, :]
    Q2 = Q2[:, int_idx1]
    S2 = Q[int_idx1, :]
    S2 = S2[:, int_idx2]

    tfixation_est = np.linalg.inv((np.eye(np.shape(Q2)[0]) - Q2))
    p_est = tfixation_est @ S2

    int_idx3 = np.array(n_mut)[1:-1]
    int_idx3 = int_idx3[int_idx1] == 1
    int_idx3 = np.where(int_idx3)[0]
    pest_mat = p_est[int_idx3, :]
    pest_mat = pest_mat[:, 1:]

    return np.array([pfixation[idx, 1] @ vec_norm, (np.sum(pest_mat, 1) @ vec_norm)])
    #return np.array([pfixation[idx, 1] @ vec_norm, np.sum(tfixation[idx, :], 1) @ vec_norm, np.sum(ctfixation[idx, :], 1) @ vec_norm])


def pfix_intersect_detour(n1, n2, a):
    def delta(f):
        f = f[0]
        d = p_t_tc_fix_detour(n1, n2, f, a)[0] - pfix(f, a, n1 + n2)
        return d
    s = fsolve(delta, [1]) - 1
    return s


##### PA Star Pfix and Pest #####

def r2(a, x):
    if a == 0.5:
        return 1
    else:
        return ((2 * a - 1) ** -2) / (sp.Rational(1, 2) * ((2 * a - 1) ** -2 - 1) + x) - 1

def pa_star(n, m):
    G = nx.complete_graph(m)
    for i in range(n):
        for j in range(m):
            G.add_edge(i+m, j)
    return G

def pfix_full_pa_star(n, m, f, a, prec=500):
    n = syp.Rational(n)
    m = syp.Rational(m)
    f = syp.Rational(f)
    a = syp.Rational(a)

    N = n + m
    vs = (n + 1) * (m + 1)

    M = sp.Matrix(np.zeros([vs, vs]).tolist())
    n_mut = sp.Matrix(np.zeros(vs).tolist())
    p_mut = sp.Matrix(np.zeros(vs - 2).tolist())
    for i in range(vs):
        n_mut[i] = i % (n + 1) + np.floor(i / (n + 1))
        if i % (n + 1) == 1 and np.floor(i / (n + 1)) == 0:
            p_mut[i - 1] = n / N
        elif i % (n + 1) == 0 and np.floor(i / (n + 1)) == 1:
            p_mut[i - 1] = m / N

        for j in range(vs):
            n_start = i % (n + 1)
            m_start = np.floor(i / (n + 1))

            n_end = j % (n + 1)
            m_end = np.floor(j / (n + 1))

            if n_end - n_start == 1 and m_end - m_start == 0:
                fit = f * r2(a, (n_start + m_start) / N)
                z = (N - (n_start + m_start) + fit * (n_start + m_start)) * (N - 1)
                pt = fit * m_start * (n - n_start)
                M[i, j] = (pt / z).evalf(prec)

            elif n_end - n_start == -1 and m_end - m_start == 0:
                fit = f * r2(a, (n_start + m_start) / N)
                z = (N - (n_start + m_start) + fit * (n_start + m_start)) * (N - 1)
                pt = (m - m_start) * n_start
                M[i, j] = (pt / z).evalf(prec)

            elif n_end - n_start == 0 and m_end - m_start == 1:
                fit = f * r2(a, (n_start + m_start) / N)
                z = (N - (n_start + m_start) + fit * (n_start + m_start))
                pt = fit * m_start * (m - m_start) / (N - 1) + fit * n_start * (m - m_start) / m
                M[i, j] = (pt / z).evalf(prec)

            elif n_end - n_start == 0 and m_end - m_start == -1:
                fit = f * r2(a, (n_start + m_start) / N)
                z = (N - (n_start + m_start) + fit * (n_start + m_start))
                pt = (m - m_start) * m_start / (N - 1) + (n - n_start) * m_start / m
                M[i, j] = (pt / z).evalf(prec)

    z = sp.Matrix(np.zeros(vs).tolist())
    for i in range(vs):
        sum_inner = sp.Rational(0, 1)
        for j in range(vs):
            if M[i, j] > 0:
                sum_inner += M[i, j]
        z[i] = sum_inner

    for i in range(vs):
        M[i, i] = sp.Rational(1, 1).evalf(prec) - z[i]

    # Fixation probability
    tmp_idx = list(range(1, vs - 1))
    Q = M[tmp_idx, tmp_idx]
    S = sp.Matrix([M[tmp_idx, 0].T, M[tmp_idx, -1].T]).T
    #p_absorb = (sp.eye(sp.Rational(vs) - 2) - Q).inv() * S
    p_absorb = (sp.eye(sp.Rational(vs, 1) - 2) - Q).LUsolve(S)
    int_idx = np.array(n_mut[tmp_idx]) == 1
    int_idx = list(np.where(int_idx)[0])
    pfix_final = p_absorb[int_idx, 1].T * sp.Matrix(p_mut[int_idx])

    # Establishment probability
    M_mod = M[tmp_idx, tmp_idx]
    M_mod2 = M[tmp_idx, list(range(vs))]

    int_idx1 = np.array(n_mut[tmp_idx]) < int(N / 2)
    int_idx1 = list(np.where(int_idx1)[0])

    int_idx2 = np.array(n_mut[tmp_idx]) == int(N / 2)
    int_idx2 = list(np.where(int_idx2)[0])

    Q = M_mod[int_idx1, int_idx1]
    S = M_mod2[int_idx1, 0].row_join(M_mod[int_idx1, int_idx2])
    p_absorb = (sp.eye(len(int_idx1)) - Q).LUsolve(S)

    int_idx3 = sp.Matrix(n_mut[tmp_idx])
    int_idx3 = np.array(int_idx3[int_idx1]) == 1
    int_idx3 = list(np.where(int_idx3)[0])
    pest_mat = p_absorb[int_idx3, list(range(1, len(int_idx2)+1))]

    return np.array([pfix_final[0, 0].evalf(10), (np.sum(pest_mat, 1) @ np.array(p_mut[int_idx])).evalf(10)])


def pfix_full_pa_star_diff(n, m, f, _a, D, prec=500):
    G = pa_star(n, m)

    n = syp.Rational(n)
    m = syp.Rational(m)
    f = syp.Rational(f)
    a = syp.Rational(_a)

    N = n + m
    vs = (n + 1) * (m + 1)

    M = sp.Matrix(np.zeros([vs, vs]).tolist())
    n_mut = sp.Matrix(np.zeros(vs).tolist())
    p_mut = sp.Matrix(np.zeros(vs - 2).tolist())
    for i in range(vs):
        n_mut[i] = i % (n + 1) + np.floor(i / (n + 1))
        if i % (n + 1) == 1 and np.floor(i / (n + 1)) == 0:
            p_mut[i - 1] = n / N
        elif i % (n + 1) == 0 and np.floor(i / (n + 1)) == 1:
            p_mut[i - 1] = m / N

        for j in range(vs):
            n_start = i % (n + 1)
            m_start = np.floor(i / (n + 1))

            n_end = j % (n + 1)
            m_end = np.floor(j / (n + 1))

            fit = network_fit(G,
                                  np.array([1] * int(m_start) + [0] * int(m - m_start) + [1] * int(n_start) + [0] * int(n - n_start)),
                                  np.array([[_a, 1 - _a], [1 - _a, _a]]),
                                  [0.5, 0.5], [D, D])

            fit_mut_m = syp.Rational(f * fit[0])
            fit_wt_m = syp.Rational(fit[int(m_start)])
            fit_mut_n = syp.Rational(f * fit[int(m)])
            fit_wt_n = syp.Rational(fit[-1])

            z = fit_mut_n * n_start + fit_wt_n * (n - n_start) + fit_mut_m * m_start + fit_wt_m * (m - m_start)

            if n_end - n_start == 1 and m_end - m_start == 0:
                pt = fit_mut_m * m_start * (n - n_start) / (N - 1)
                M[i, j] = (pt / z).evalf(prec)

            elif n_end - n_start == -1 and m_end - m_start == 0:
                pt = fit_wt_m * (m - m_start) * n_start / (N - 1)
                M[i, j] = (pt / z).evalf(prec)

            elif n_end - n_start == 0 and m_end - m_start == 1:
                pt = fit_mut_m * m_start * (m - m_start) / (N - 1) + fit_mut_n * n_start * (m - m_start) / m
                M[i, j] = (pt / z).evalf(prec)

            elif n_end - n_start == 0 and m_end - m_start == -1:
                pt = fit_wt_m * (m - m_start) * m_start / (N - 1) + fit_wt_n * (n - n_start) * m_start / m
                M[i, j] = (pt / z).evalf(prec)

    z = sp.Matrix(np.zeros(vs).tolist())
    for i in range(vs):
        sum_inner = sp.Rational(0, 1)
        for j in range(vs):
            if M[i, j] > 0:
                sum_inner += M[i, j]
        z[i] = sum_inner

    for i in range(vs):
        M[i, i] = sp.Rational(1, 1).evalf(prec) - z[i]

    # Fixation probability
    tmp_idx = list(range(1, vs - 1))
    Q = M[tmp_idx, tmp_idx]
    S = sp.Matrix([M[tmp_idx, 0].T, M[tmp_idx, -1].T]).T
    #p_absorb = (sp.eye(sp.Rational(vs) - 2) - Q).inv() * S
    p_absorb = (sp.eye(sp.Rational(vs, 1) - 2) - Q).LUsolve(S)
    int_idx = np.array(n_mut[tmp_idx]) == 1
    int_idx = list(np.where(int_idx)[0])
    pfix_final = p_absorb[int_idx, 1].T * sp.Matrix(p_mut[int_idx])

    # Establishment probability
    M_mod = M[tmp_idx, tmp_idx]
    M_mod2 = M[tmp_idx, list(range(vs))]

    int_idx1 = np.array(n_mut[tmp_idx]) < int(N / 2)
    int_idx1 = list(np.where(int_idx1)[0])

    int_idx2 = np.array(n_mut[tmp_idx]) == int(N / 2)
    int_idx2 = list(np.where(int_idx2)[0])

    Q = M_mod[int_idx1, int_idx1]
    S = M_mod2[int_idx1, 0].row_join(M_mod[int_idx1, int_idx2])
    p_absorb = (sp.eye(len(int_idx1)) - Q).LUsolve(S)

    int_idx3 = sp.Matrix(n_mut[tmp_idx])
    int_idx3 = np.array(int_idx3[int_idx1]) == 1
    int_idx3 = list(np.where(int_idx3)[0])
    pest_mat = p_absorb[int_idx3, list(range(1, len(int_idx2)+1))]

    return np.array([pfix_final[0, 0].evalf(10), (np.sum(pest_mat, 1) @ np.array(p_mut[int_idx])).evalf(10)])


def pfix_intersect_pa_star(n, m, a):
    def delta(f):
        f = f[0]
        d = pfix_full_pa_star(n, m, f, a)[0] - pfix(f, a, n + m)
        return d
    s = fsolve(delta, [1]) - 1
    return s


def pfix(f, alpha, N):
    phi = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N))
        phi += tmp

    return 1 / (1 + phi)


def pest(f, alpha, N):
    phi = 0
    for i in range(1, int(N / 2)):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N))
        phi += tmp

    return 1 / (1 + phi)


def pint(f, alpha, N):
    phi1 = 0
    for i in range(1, int(N / 2)):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N))
        phi1 += tmp

    phi2 = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N))
        phi2 += tmp

    return (1 + phi1) / (1 + phi2)



###### Star Pfix Functions ######

# this p_t is for well-mixed resources
def p_t(x, y, i, j, f, alpha, N):
    fit = f * est.r(alpha, (x + i) / N)

    if x == 1 and y == 1 and j - i == 1:
        return fit / ((i + 1) * fit + (N - (i + 1))) * ((N - 1) - i) / (N - 1)

    elif x == 0 and y == 0 and j - i == -1:
        return 1 / (i * fit + (N - i)) * i / (N - 1)

    elif x == 1 and y == 0 and j - i == 0:
        return ((N - 1) - i) / ((i + 1) * fit + (N - (i + 1)))

    elif x == 0 and y == 1 and j - i == 0:
        return fit * i / (i * fit + (N - i))


def pi_t(x, y, i, j, f, alpha, N):
    if x == 1 and y == 1 and j - i == 1:
        p0 = p_t(1, 1, i, j, f, alpha, N)
        p1 = p_t(1, 0, i, i, f, alpha, N)
        return p0 / (p0 + p1)

    elif x == 0 and y == 0 and j - i == -1:
        p0 = p_t(0, 0, i, j, f, alpha, N)
        p1 = p_t(0, 1, i, i, f, alpha, N)
        return p0 / (p0 + p1)

    elif x == 1 and y == 0 and j - i == 0:
        p0 = p_t(1, 1, i, j + 1, f, alpha, N)
        p1 = p_t(1, 0, i, j, f, alpha, N)
        return p1 / (p0 + p1)

    elif x == 0 and y == 1 and j - i == 0:
        p0 = p_t(0, 0, i, j - 1, f, alpha, N)
        p1 = p_t(0, 1, i, i, f, alpha, N)
        return p1 / (p0 + p1)


def A(l, m, f, alpha, N):
    out = 1
    for j in range(l, m):
        prod_inner = pi_t(1, 0, j, j, f, alpha, N)
        for k in range(l, j + 1):
            prod_inner *= np.round(pi_t(0, 0, k, k - 1, f, alpha, N) / pi_t(1, 1, k, k + 1, f, alpha, N), 10)

        out += prod_inner

    return out


def p_aa(i, f, alpha, N):
    return A(1, i, f, alpha, N) / A(1, N - 1, f, alpha, N)


def p_aa_0(f, alpha, N):
    return pi_t(1, 1, 0, 1, f, alpha, N) / A(1, N - 1, f, alpha, N)


def p_ab(i, f, alpha, N):
    out = 0
    for j in range(1, i + 1):
        prod_inner = pi_t(0, 1, j, j, f, alpha, N) * p_aa(j, f, alpha, N)
        for k in range(j + 1, i + 1):
            prod_inner *= pi_t(0, 0, k, k - 1, f, alpha, N)

        out += prod_inner

    return out


def pfix_star(f, alpha, N):
    return 1/N * ((N - 1) * p_ab(1, f, alpha, N) + p_aa_0(f, alpha, N))


def pint_star(f, alpha, N):
    return p_aa(int(N/2) - 1, f, alpha, N)


def pfix_intersect_star(N, a):
    def delta(f):
        d = pfix(f, a, N) - pfix_star(f, a, N)
        return d
    s = fsolve(delta, [1]) - 1
    return s


def pfix_star_approx(f, alpha, N):
    phi = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N)) ** 2
        phi += tmp

    return 1 / (1 + phi)


def pest_star_approx(f, alpha, N):
    phi = 0
    for i in range(1, int(N / 2)):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N)) ** 2
        phi += tmp

    return 1 / (1 + phi)


def pint_star_approx(f, alpha, N):
    phi1 = 0
    for i in range(1, int(N / 2)):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N)) ** 2
        phi1 += tmp

    phi2 = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * est.r(alpha, j / N)) ** 2
        phi2 += tmp

    return (1 + phi1) / (1 + phi2)


def pfix_intersect_star_approx(N, a):
    def delta(f):
        d = pfix(f, a, N) - pfix_star_approx(f, a, N)
        return d
    s = fsolve(delta, [1]) - 1
    return s


###### Diffusion Star Pfix Functions ######
def fitness_compute(f, alpha, D, N):
    G = nx.star_graph(N - 1)
    fit_mat = np.zeros([2, N, 4])
    for x in [0, 1]:
        for i in range(N):
            fits = network_fit(G, np.array([x] + [1] * i + [0] * (N - i - 1)), [[alpha, 1 - alpha], [1 - alpha, alpha]],
                               [0.5, 0.5], [D, D])
            fit_mat[x, i, 0] = f * fits[0]
            fit_mat[x, i, 1] = fits[0]
            fit_mat[x, i, 2] = f * fits[1]
            fit_mat[x, i, 3] = fits[-1]

    return fit_mat


def p_t_diff(x, y, i, j, fit_mat, N):

    fits = fit_mat[x, i, :]
    fit_c_1 = fits[0]
    fit_c_0 = fits[1]
    fit_l_1 = fits[2]
    fit_l_0 = fits[3]

    if x == 1 and y == 1 and j - i == 1:
        return fit_c_1 / (fit_c_1 + i * fit_l_1 + (N - i - 1) * fit_l_0) * ((N - 1) - i) / (N - 1)

    elif x == 0 and y == 0 and j - i == -1:
        return fit_c_0 / (fit_c_0 + i * fit_l_1 + (N - i - 1) * fit_l_0) * i / (N - 1)

    elif x == 1 and y == 0 and j - i == 0:
        return (N - i - 1) * fit_l_0 / (fit_c_1 + i * fit_l_1 + (N - i - 1) * fit_l_0)

    elif x == 0 and y == 1 and j - i == 0:
        return i * fit_l_1 / (fit_c_0 + i * fit_l_1 + (N - i - 1) * fit_l_0)


def pi_t_diff(x, y, i, j, fit_mat, N):
    if x == 1 and y == 1 and j - i == 1:
        p0 = p_t_diff(1, 1, i, j, fit_mat, N)
        p1 = p_t_diff(1, 0, i, i, fit_mat, N)
        return p0 / (p0 + p1)

    elif x == 0 and y == 0 and j - i == -1:
        p0 = p_t_diff(0, 0, i, j, fit_mat, N)
        p1 = p_t_diff(0, 1, i, i, fit_mat, N)
        return p0 / (p0 + p1)

    elif x == 1 and y == 0 and j - i == 0:
        p0 = p_t_diff(1, 1, i, j + 1, fit_mat, N)
        p1 = p_t_diff(1, 0, i, j, fit_mat, N)
        return p1 / (p0 + p1)

    elif x == 0 and y == 1 and j - i == 0:
        p0 = p_t_diff(0, 0, i, j - 1, fit_mat, N)
        p1 = p_t_diff(0, 1, i, i, fit_mat, N)
        return p1 / (p0 + p1)


def A_diff(l, m, fit_mat, N):
    out = 1
    for j in range(l, m):
        prod_inner = pi_t_diff(1, 0, j, j, fit_mat, N)
        for k in range(l, j + 1):
            prod_inner *= np.round(pi_t_diff(0, 0, k, k - 1, fit_mat, N) / pi_t_diff(1, 1, k, k + 1, fit_mat, N), 10)

        out += prod_inner

    return out


def p_aa_diff(i, fit_mat, N):
    return A_diff(1, i, fit_mat, N) / A_diff(1, N - 1, fit_mat, N)


def p_aa_0_diff(fit_mat, N):
    return pi_t_diff(1, 1, 0, 1, fit_mat, N) / A_diff(1, N - 1, fit_mat, N)


def p_ab_diff(i, fit_mat, N):
    out = 0
    for j in range(1, i + 1):
        prod_inner = pi_t_diff(0, 1, j, j, fit_mat, N) * p_aa_diff(j, fit_mat, N)
        for k in range(j + 1, i + 1):
            prod_inner *= pi_t_diff(0, 0, k, k - 1, fit_mat, N)

        out += prod_inner

    return out


def pfix_star_diff(fit_mat, N):
    return 1/N * ((N - 1) * p_ab_diff(1, fit_mat, N) + p_aa_0_diff(fit_mat, N))


def pfix_intersect_star_diff(N, a, D):
    def delta(f):
        fit_mat = fitness_compute(f, a, D, N)
        d = pfix(f, a, N) - pfix_star_diff(fit_mat, N)
        return d
    s = fsolve(delta, [1]) - 1
    return s


###### Generalized WM Pfix Functions ######
#alpha_mat: [[r1 wt, r1 mut] , [r2 wt, r2 mut]]

def r_gen(alpha_mat, S_mat, x):
    c1 = S_mat[0] / ((1 - x) * alpha_mat[0][0] + x * alpha_mat[0][1])
    c2 = S_mat[1] / ((1 - x) * alpha_mat[1][0] + x * alpha_mat[1][1])
    f_wt = c1 * alpha_mat[0][0] + c2 * alpha_mat[1][0]
    f_mut = c1 * alpha_mat[0][1] + c2 * alpha_mat[1][1]
    r = f_mut / f_wt
    return r

def pfix_gen(f, alpha_mat, S_mat, N):
    phi = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= 1 / (f * r_gen(alpha_mat, S_mat, j / N))
        phi += tmp

    return 1 / (1 + phi)


###### Generalized Star Pfix Functions ######
#alpha_mat: [[r1 s1, r1 s2] , [r2 s1, r2 s2]]

def fitness_compute_gen(f, alpha_mat, S_mat, D_mat, N):
    G = nx.star_graph(N - 1)
    fit_mat = np.zeros([2, N, 4])
    for x in [0, 1]:
        for i in range(N):
            fits = network_fit(G, np.array([x] + [1] * i + [0] * (N - i - 1)), alpha_mat,
                               S_mat, D_mat)
            fit_mat[x, i, 0] = f * fits[0]
            fit_mat[x, i, 1] = fits[0]
            fit_mat[x, i, 2] = f * fits[1]
            fit_mat[x, i, 3] = fits[-1]

    return fit_mat


def p_t_gen(x, y, i, j, fit_mat, N):

    fits = fit_mat[x, i, :]
    fit_c_1 = fits[0]
    fit_c_0 = fits[1]
    fit_l_1 = fits[2]
    fit_l_0 = fits[3]

    if x == 1 and y == 1 and j - i == 1:
        return fit_c_1 / (fit_c_1 + i * fit_l_1 + (N - i - 1) * fit_l_0) * ((N - 1) - i) / (N - 1)

    elif x == 0 and y == 0 and j - i == -1:
        return fit_c_0 / (fit_c_0 + i * fit_l_1 + (N - i - 1) * fit_l_0) * i / (N - 1)

    elif x == 1 and y == 0 and j - i == 0:
        return (N - i - 1) * fit_l_0 / (fit_c_1 + i * fit_l_1 + (N - i - 1) * fit_l_0)

    elif x == 0 and y == 1 and j - i == 0:
        return i * fit_l_1 / (fit_c_0 + i * fit_l_1 + (N - i - 1) * fit_l_0)


def pi_t_gen(x, y, i, j, fit_mat, N):
    if x == 1 and y == 1 and j - i == 1:
        p0 = p_t_gen(1, 1, i, j, fit_mat, N)
        p1 = p_t_gen(1, 0, i, i, fit_mat, N)
        return p0 / (p0 + p1)

    elif x == 0 and y == 0 and j - i == -1:
        p0 = p_t_gen(0, 0, i, j, fit_mat, N)
        p1 = p_t_gen(0, 1, i, i, fit_mat, N)
        return p0 / (p0 + p1)

    elif x == 1 and y == 0 and j - i == 0:
        p0 = p_t_gen(1, 1, i, j + 1, fit_mat, N)
        p1 = p_t_gen(1, 0, i, j, fit_mat, N)
        return p1 / (p0 + p1)

    elif x == 0 and y == 1 and j - i == 0:
        p0 = p_t_gen(0, 0, i, j - 1, fit_mat, N)
        p1 = p_t_gen(0, 1, i, i, fit_mat, N)
        return p1 / (p0 + p1)


def A_gen(l, m, fit_mat, N):
    out = 1
    for j in range(l, m):
        prod_inner = pi_t_gen(1, 0, j, j, fit_mat, N)
        for k in range(l, j + 1):
            prod_inner *= np.round(pi_t_gen(0, 0, k, k - 1, fit_mat, N) / pi_t_gen(1, 1, k, k + 1, fit_mat, N), 10)

        out += prod_inner

    return out


def p_aa_gen(i, fit_mat, N):
    return A_gen(1, i, fit_mat, N) / A_gen(1, N - 1, fit_mat, N)


def p_aa_0_gen(fit_mat, N):
    return pi_t_gen(1, 1, 0, 1, fit_mat, N) / A_gen(1, N - 1, fit_mat, N)


def p_ab_gen(i, fit_mat, N):
    out = 0
    for j in range(1, i + 1):
        prod_inner = pi_t_gen(0, 1, j, j, fit_mat, N) * p_aa_gen(j, fit_mat, N)
        for k in range(j + 1, i + 1):
            prod_inner *= pi_t_gen(0, 0, k, k - 1, fit_mat, N)

        out += prod_inner

    return out


def pfix_star_gen(fit_mat, N):
    return 1/N * ((N - 1) * p_ab_gen(1, fit_mat, N) + p_aa_0_gen(fit_mat, N))
