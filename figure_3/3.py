import numpy as np
import matplotlib.pyplot as plt
import amp_and_acc as amp
import pfix_analytic as p

p_list = []
pest_list = []

amp_list = []

for i in range(1, 50):
    out = p.pfix_full_pa_star(50 - i, i, 1 - 0.012, 0.8)
    p_list.append(out[0])
    pest_list.append(out[1])
    
    G = p.pa_star(50 - i, i)
    amp_list.append(amp.amplification_and_acceleration(G)[0])
    

plt.figure()
plt.scatter(pest_list, np.divide(p_list, pest_list), c=p_list)

plt.figure()
plt.scatter(amp_list, p_list, c=p_list)

plt.show()