import numpy as np
import networkx as nx


def approx3(G, a):
    nodes = list(G)
    N = len(nodes)
    adj = nx.adjacency_matrix(G)
    D_inv = np.linalg.inv(np.diag(list(dict(nx.degree(G)).values())))
    b = (np.linalg.inv(np.diag(list(dict(nx.degree(G)).values()))) @ nx.adjacency_matrix(G)).T @ np.ones(N)

    p_i_ij = (r(a, 1/N) / (N + r(a, 1/N) - 1)) * D_inv @ adj
    p_ij_i = np.zeros([N, N])
    p_i_i = 1 - (1 / (N + r(a, 1/N) - 1)) * (r(a, 1/N) + b)
    p_ij_3 = np.zeros([N, N])
    p_ij_ij = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            p_ij_i[i, j] = adj[i, j] * (1 / (N + 2 * (r(a, 2/N) - 1))) * (b[j] - D_inv[i, i])
            p_ij_3[i, j] = (r(a, 2/N) / (N + 2 * (r(a, 2/N) - 1))) * (2 - D_inv[i, i] - D_inv[j, j])

    for i in range(N):
        for j in range(N):
            p_ij_ij[i, j] = adj[i, j] * (1 - p_ij_i[i, j] - p_ij_i[j, i] - p_ij_3[i, j])

    U = np.zeros([N, N])
    V = np.zeros(N)

    for i in range(N):
        tmp1 = 0
        tmp2 = 0
        for j in range(N):
            if i != j:
                tmp1 += ((1 - p_i_i[i]) ** -1) * p_i_ij[i, j] * ((1 - p_ij_ij[i, j]) ** -1) * p_ij_i[i, j]
                tmp2 += ((1 - p_i_i[i]) ** -1) * p_i_ij[i, j] * ((1 - p_ij_ij[i, j]) ** -1) * p_ij_3[i, j]
                U[i, j] = ((1 - p_i_i[i]) ** -1) * p_i_ij[i, j] * ((1 - p_ij_ij[i, j]) ** -1) * p_ij_i[j, i]

        U[i, i] = tmp1
        V[i] = tmp2

    pi = np.linalg.solve(np.eye(N)-U, V)
    return pi


def approx3_a(G, a):
    nodes = np.array(list(G))
    srt_idx = np.argsort(nodes).astype(int)
    nodes = nodes[srt_idx]

    factor = (np.linalg.inv(np.diag(list(dict(nx.degree(G)).values()))) @ nx.adjacency_matrix(G)).T @ np.ones(100)
    factor = factor[srt_idx]
    pi1 = (1 / (1 + (r(a, 1 / 100) ** -1) * factor))

    pi2 = np.zeros(len(nodes))

    for node in nodes:
        di = G.degree(node)
        Ri = factor[node]

        nn = nx.neighbors(G, node)
        for n in nn:
            dj = G.degree(node)
            Rj = factor[node]

            pi2[node] += (2 - (di**-1 + dj**-1)) / (2 - (1 + r(a, 1 / 100) ** -1) * (di**-1 + dj**-1) + (r(a, 1 / 100) ** -1) * (Ri + Rj))

        pi2[node] = pi1[node] * pi2[node] / di

    return pi2


def r(a, x):
    if a == 0.5:
        return 1
    else:
        return ((2 * a - 1) ** -2) / (0.5 * ((2 * a - 1) ** -2 - 1) + x) - 1


def wm_p_est(a, N):
    p = 1 / (1 + 1/r(a, 1/N) * (1 - 1/np.log(1/r(a, 1/N))))
    return p


def wm_tfix(a, N):
    t = np.sqrt(np.pi/(2 * N)) * np.exp(0.5 * N * (2 * a - 1) ** 2) / ((2 * a - 1) ** 3)
    return t


def wm_p_est_s(s, a, N):
    p = 1 / (1 + 1/((1 + s) * r(a, 1/N)) * (1 - 1/np.log(1/((1 + s) * r(a, 1/N)))))
    return p


def wm_tfix_s(s, a, N):
    if s > 0:
        rho = (1 - (1 + s) ** (-N))
    elif s < 0:
        rho = (1 + s) ** (N)
    else:
        rho = 0.5

    phi = 0.5 + (s / (2 + s)) * 0.5 * ((2 * a - 1) ** -2)
    w = 2 * a - 1
    t = rho \
        * np.sqrt(2 * np.pi * (1 + s) ** 3 / (N * (w ** 6) * ((2 + s) ** 6) * (phi ** 2) * ((1 - phi) ** 4))) \
        * np.exp((N / (2 * (1 + s))) * (w * (2 + s) * (1 - phi)) ** 2)

    return t


def star_p_est_s(s, a, N):
    p = 1 / (1 + 1/((1 + s) * r(a, 1/N)) ** 2 * (1 - 1/np.log(1/((1 + s) * r(a, 1/N)) ** 2)))
    return p


def star_tfix_s(s, a, N):
    if s > 0:
        rho = (1 - (1 + s) ** (-2 * N))
    elif s < 0:
        rho = (1 + s) ** (2 * N)
    else:
        rho = 0.5

    phi = 0.5 + (s / (2 + s)) * 0.5 * ((2 * a - 1) ** -2)
    w = 2 * a - 1
    t = N * 2 ** (-3/2)\
        * rho \
        * np.sqrt(2 * np.pi * (1 + s) ** 3 / (N * (w ** 6) * ((2 + s) ** 6) * (phi ** 2) * ((1 - phi) ** 4))) \
        * np.exp(2 * (N / (2 * (1 + s))) * (w * (2 + s) * (1 - phi)) ** 2)

    return t


def strong_g(G):
    degree = list(dict(nx.degree(G)).values())
    factor = 1 / ((np.linalg.inv(np.diag(degree)) @ nx.adjacency_matrix(G)).T @ np.ones(len(degree)))
    g = np.log(factor) / np.log(len(degree)) + 1
    return g


def G_p_est_s(g, s, a, N):
    p = (1 - 1 / (((1 + s) * r(a, 1/N)) ** g)) #/ (1 - 1 / (((1 + s) * r(a, 1 / N)) ** (50 * g)))
    return p


def G_rho_s(g, s, N):
    rho = 1 / (1 + (1 + s) ** (-g * (N)))

    return rho


def G_tfix_s(g1, g2, s, a, N):
    if s > 0:
        rho = (1 - (1 + s) ** (-g2 * N))
    elif s < 0:
        rho = (1 + s) ** (g2 * N)
    else:
        rho = 0.5

    phi = 0.5 + (s / (2 + s)) * 0.5 * ((2 * a - 1) ** -2)
    w = 2 * a - 1
    t = g1 * g2 ** (-3/2)\
        * rho \
        * np.sqrt(2 * np.pi * (1 + s) ** 3 / (N * (w ** 6) * ((2 + s) ** 6) * (phi ** 2) * ((1 - phi) ** 4))) \
        * np.exp(g2 * (N / (2 * (1 + s))) * (w * (2 + s) * (1 - phi)) ** 2)

    return t


def G_p_est_exact(g, a, f, N):
    if a == 0.5:
        x_st = int(N / 2) # quasi-equilibrium case
    else:
        x_st = int(np.round(N * (0.5 + 0.5 * (f - 1) / (f + 1) * (2 * a - 1) ** -2)))

    phi = 0
    for i in range(1, x_st):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= (f * r(a, j/N)) ** (-g)
        phi += tmp

    return 1 / (1 + phi)


def G_p_fix_exact(g, a, f, N, init):

    phi = 0
    phi0 = 0
    for i in range(1, N):
        tmp = 1
        for j in range(1, i + 1):
            tmp *= (f * r(a, j/N)) ** (-g)
        phi += tmp

        if i == init - 1:
            phi0 = phi

    return (1 + phi0) / (1 + phi)


def G_tfix_exact(g1, g, a, f, N):
    phi = 0.5 + 0.5 * (f-1)/(f+1) * (2*a - 1)**(-2)

    t1 = 0
    for i in range(int(phi * N), N):
        for j in range(1, i+1):
            tmp = 1 / (f * r(a, j/N)**g * j * (N - j) / (N ** 2))
            for k in range(j+1, i+1):
                tmp *= 1 / (f * r(a, k/N)**g)
            t1 += tmp

    t2 = 0
    for i in range(1, N):
        for j in range(1, i + 1):
            tmp = 1 / (f * r(a, j / N)**g * j * (N - j) / (N ** 2))
            for k in range(j + 1, i + 1):
                tmp *= 1 / (f * r(a, k / N)**g)
            t2 += tmp

    p1 = 0
    for j in range(int(phi * N), N):
        tmp = 1
        for k in range(1, j+1):
            tmp *= 1 / (f * r(a, k / N)**g)
        p1 += tmp

    p2 = 0
    for j in range(1, N):
        tmp = 1
        for k in range(1, j + 1):
            tmp *= 1 / (f * r(a, k / N)**g)
        p2 += tmp

    t = t1 - (p1 / (1 + p2)) * t2

    return g1 * t / N

