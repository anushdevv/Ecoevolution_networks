import numpy as np
import matplotlib.pyplot as plt
import amp_and_acc as amp
import pfix_analytic as p

p_list = []
p_est_list = []
for i in range(1, 11):
    a, b = p.pfix_full_pa_star_diff(50 - i, i, 1 - 0.01, 0.501, 10)
    p_list.append(a)
    p_est_list.append(b)
    
p_list_1 = []
p_est_list_1 = []
for i in range(1, 11):
    a, b = p.pfix_full_pa_star_diff(50 - i, i, 1 - 0.01, 0.501, 1)
    p_list_1.append(a)
    p_est_list_1.append(b)
    
p_list2 = []
p_est_list2 = []
for i in range(1, 11):
    a, b = p.pfix_full_pa_star_diff(50 - i, i, 1 - 0.01, 0.65, 10)
    p_list2.append(a)
    p_est_list2.append(b)
    
p_list2_1 = []
p_est_list2_1 = []
for i in range(1, 11):
    a, b = p.pfix_full_pa_star_diff(50 - i, i, 1 - 0.01, 0.65, 1)
    p_list2_1.append(a)
    p_est_list2_1.append(b)
    
p_list3 = []
p_est_list3 = []
for i in range(1, 11):
    a, b = p.pfix_full_pa_star_diff(50 - i, i, 1 - 0.01, 0.99, 10)
    p_list3.append(a)
    p_est_list3.append(b)
    
p_list3_1 = []
p_est_list3_1 = []
for i in range(1, 11):
    a, b = p.pfix_full_pa_star_diff(50 - i, i, 1 - 0.01, 0.99, 1)
    p_list3_1.append(a)
    p_est_list3_1.append(b)

amp_list = []
for i in range(1, 11):
    G = p.pa_star(50 - i, i)
    amp_list.append(amp.amplification_and_acceleration(G)[0])

#plt.scatter(amp_list[:10], p_list_1)
plt.scatter(amp_list[:10], p_list)
a = 0.501
s = -0.01
alphas = np.linspace(1, 2, 100)
plt.plot(alphas, 1 / 50 + alphas * (s / 2 + (2 * a - 1) ** 2 / 3))
plt.show()

#plt.scatter(amp_list[:10], p_list2_1)
plt.scatter(amp_list[:10], p_list2)
a = 0.65
s = -0.01
alphas = np.linspace(1, 2, 100)
plt.plot(alphas, 1 / 50 + alphas * (s / 2 + (2 * a - 1) ** 2 / 3))
plt.show()

#plt.scatter(amp_list[:10], p_list3_1)
plt.scatter(amp_list[:10], p_list3)
s = -0.01
alphas = np.linspace(1, 2, 10)
plt.plot(alphas, 1 / 2 + 1/4 * alphas *50*s)
plt.show()

