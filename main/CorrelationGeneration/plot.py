import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import sys
filename = sys.argv[1]
output = sys.argv[2]
limitx = float(sys.argv[3])
limity = float(sys.argv[4])
#vmin = int(sys.argv[5])
#vmax = int(sys.argv[6])
 
data = np.loadtxt(filename)
plt.figure()
plt.title(f"{filename}")

#plt.grid(which = 'both',color='w', linestyle='-', linewidth=1)
plt.imshow(data, origin = 'lower', cmap='hot', vmin = 0, extent = [-limitx,limitx,-limity,limity]) #'YlGnBu'vmin = -10, vmax = 100, extent = [-18,18,-26,26]
plt.colorbar()

plt.savefig(output)
plt.show()