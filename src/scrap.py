import numpy as np
import control as ct

des_poles = [0.1]
for p in [0.1*(np.cos(p) + np.sin(p)*1j) for p in np.linspace(0.1, 2*np.pi - 0.1, 12)]:
    des_poles.append(p)


print(des_poles)
