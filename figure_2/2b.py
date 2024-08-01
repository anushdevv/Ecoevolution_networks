import numpy as np
import matplotlib.pyplot as plt
import amp_and_acc as amp
import pfix_analytic as p

p_list = []
p_list2 = []
p_star_list = []
p_star_list2 = []

for s in np.linspace(-0.2, 0.2, 100):
    p_list.append(p.pest(1 + s, 0.8, 500))
    p_list2.append(p.pest(1 + s, 1-1e-10, 500))
    
    p_star_list.append(p.pfix_star(1 + s, 0.8, 500) / p.pint_star(1 + s, 0.8, 500))
    p_star_list2.append(p.pfix_star(1 + s, 1-1e-10, 500) / p.pint_star(1 + s, 1-1e-10, 500))
    
plt.plot(np.linspace(-0.2, 0.2, 100), p_list)
plt.plot(np.linspace(-0.2, 0.2, 100), p_list2)
plt.plot(np.linspace(-0.2, 0.2, 100), p_star_list)
plt.plot(np.linspace(-0.2, 0.2, 100), p_star_list2)
    
plt.show()