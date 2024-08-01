import numpy as np
import matplotlib.pyplot as plt
import amp_and_acc as amp
import pfix_analytic as p

p_list = []
p_list2 = []
p_star_list = []
p_star_list2 = []

for a in np.linspace(0.5 + 1e-10, 1 - 1e-10, 100):
    p_list.append(p.pint(1 + 0.001, a, 500))
    p_list2.append(p.pint(1 - 0.001, a, 500))
    
    p_star_list.append(p.pint_star(1 + 0.001, a, 500))
    p_star_list2.append(p.pint_star(1 - 0.001, a, 500))
    
plt.plot(np.linspace(0.5 + 1e-10, 1 - 1e-10, 100), p_list)
plt.plot(np.linspace(0.5 + 1e-10, 1 - 1e-10, 100), p_list2)
plt.plot(np.linspace(0.5 + 1e-10, 1 - 1e-10, 100), p_star_list)
plt.plot(np.linspace(0.5 + 1e-10, 1 - 1e-10, 100), p_star_list2)

plt.show()