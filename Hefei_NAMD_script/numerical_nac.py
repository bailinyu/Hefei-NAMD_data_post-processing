# coding=utf-8
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
path = './NATXT'

natxt = np.loadtxt(path)
nac = np.average(np.abs(natxt), axis=0)
n = int(np.sqrt(len(nac)))
nac.resize(n,n)

h = 4.1356676969e-15 #约化普朗克常数，单位eV·s
dt = 1e-15 #时间差值，单位fs
p = math.pi
nac_e = h/(4*dt*p) #转换位eV的能量
nac = nac * nac_e
nac = nac * 1e+3 #转换为meV


print(nac)
plt.imshow(nac,vmin=0,vmax=1, cmap='YlGnBu', origin='lower')
plt.colorbar()
#YlGnBu
plt.title('NA Coupling',size=16)

plt.yticks([0,1],['$VBM$','$CBM$'],size=14)
plt.xticks([0,1],['$VBM$','$CBM$'],size=14,rotation = 30)
#
plt.tight_layout()
plt.savefig('nac.png',dpi=600)
