import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

path = './NATXT'

natxt = np.loadtxt(path)
nac = np.average(np.abs(natxt), axis=0)
n = int(np.sqrt(len(nac)))
nac.resize(n,n)

plt.imshow(nac, cmap='bwr', origin='lower')
plt.colorbar()

plt.title('NA Coupling',size=16)

plt.yticks([0,1,2],['$VBM$','$CBM$','$CB$'],size=16)
plt.xticks([0,1,2],['$VBM$','$CBM$','$CB$'],size=16)

plt.tight_layout()
plt.savefig('COUPLE.png')
