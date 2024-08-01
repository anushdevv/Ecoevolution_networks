import numpy as np
import matplotlib.pyplot as plt
import amp_and_acc as amp
import pfix_analytic as p

p_list = []
pest_list = []

p_list2 = []
pest_list2 = []

amp_list = []

for i in range(1, 50):
    out = p.pfix_full_pa_star(50 - i, i, 1 - 0.01, 0.8)
    p_list.append(out[0])
    pest_list.append(out[1])
    
    out = p.pfix_full_pa_star(50 - i, i, 1 - 0.01, 0.99)
    p_list2.append(out[0])
    pest_list2.append(out[1])
    
    G = p.pa_star(50 - i, i)
    amp_list.append(amp.amplification_and_acceleration(G)[0])
    

plt.figure()
plt.scatter(amp_list, pest_list)
plt.scatter(amp_list, pest_list2)

plt.figure()
plt.scatter(amp_list, np.divide(p_list, pest_list))
plt.scatter(amp_list, np.divide(p_list2, pest_list2))

plt.figure()
plt.scatter(amp_list, p_list)
plt.scatter(amp_list, p_list2)

plt.show()