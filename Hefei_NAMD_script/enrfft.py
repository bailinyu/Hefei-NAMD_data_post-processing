#!/usr/bin/env python
############################################################
bottom_band = 135
top_band = 137
import os, re
import numpy as np
from glob import glob

import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
############################################################
def WeightFromPro(infile='PROCAR', whichAtom=None, spd=None):
    """
    Contribution of selected atoms to the each KS orbital
    """

    print(infile)
    assert os.path.isfile(infile), '%s cannot be found!' % infile
    FileContents = [line for line in open(infile) if line.strip()]

    # when the band number is too large, there will be no space between ";" and
    # the actual band number. A bug found by Homlee Guo.
    # Here, #kpts, #bands and #ions are all integers
    nkpts, nbands, nions = [int(xx) for xx in re.sub('[^0-9]', ' ', FileContents[1]).split()]

    if spd:
        Weights = np.asarray([line.split()[1:-1] for line in FileContents
                              if not re.search('[a-zA-Z]', line)], dtype=float)
        Weights = np.sum(Weights[:,spd], axis=1)
    else:
        Weights = np.asarray([line.split()[-1] for line in FileContents
                              if not re.search('[a-zA-Z]', line)], dtype=float)

    nspin = Weights.shape[0] // (nkpts * nbands * nions)
    Weights.resize(nspin, nkpts, nbands, nions)

    Energies = np.asarray([line.split()[-4] for line in FileContents
                            if 'occ.' in line], dtype=float)
    Energies.resize(nspin, nkpts, nbands)

    if whichAtom is None:
        return Energies, np.sum(Weights, axis=-1)
    else:
        # whichAtom = [xx - 1 for xx in whichAtom]
        return Energies, np.sum(Weights[:,:,:,whichAtom], axis=-1)

def parallel_wht(runDirs, whichAtoms, nproc=None):
    '''
    calculate localization of some designated in parallel.
    '''
    import multiprocessing
    nproc = multiprocessing.cpu_count() if nproc is None else nproc
    pool = multiprocessing.Pool(processes=nproc)

    results = []
    for rd in runDirs:
        res = pool.apply_async(WeightFromPro, (rd + '/PROCAR', whichAtoms, None,))
        results.append(res)

    enr = []
    wht = []
    for ii in range(len(results)):
        tmp_enr, tmp_wht = results[ii].get()
        enr.append(tmp_enr)
        wht.append(tmp_wht)

    return np.array(enr), np.array(wht)

############################################################
# calculate spatial localization
############################################################
nsw     = 2000
dt      = 1.0
nproc   = 8
prefix  = '../run'
runDirs = [prefix + '/{:04d}'.format(ii + 1) for ii in range(nsw)]
# which spin, index starting from 0
whichS  = 0
# which k-point, index starting from 0
whichK  = 0
# which atoms, index starting from 0
whichA  = np.arange(0) + 108
# whichB  = range(54)
Alabel  = r'MoS$_2$'
Blabel  = r'WS$_2$'

if os.path.isfile('all_wht.npy'):
    Wht = np.load('all_wht.npy')
    Enr = np.load('all_en.npy')
else:
    # for gamma point version, no-spin
    Enr, Wht = parallel_wht(runDirs, whichA, nproc=nproc)
    Enr = Enr[:, whichS,whichK, :]
    Wht = Wht[:, whichS,whichK, :]

    # Enr, Wht1 = parallel_wht(runDirs, whichA, nproc=nproc)
    # Enr, Wht2 = parallel_wht(runDirs, whichB, nproc=nproc)
    # Enr = Enr[:, whichS,whichK, :]
    # Wht1 = Wht1[:, whichS,whichK, :]
    # Wht2 = Wht2[:, whichS,whichK, :]
    # Wht = Wht1 / (Wht1 + Wht2)

    np.save('all_wht.npy', Wht)
    np.save('all_en.npy', Enr)
enrtxt = Enr[:, bottom_band:top_band]
np.savetxt('ENRTXT', enrtxt)

from scipy.fftpack import fftfreq, fft, ifft

# Two-column data
et       = enrtxt[:,1]
# subtract the average to make the amplitude of zero-frequency 0
et      -= np.average(et)
# No. of data points
nsw      = et.size
# the time step in unit of fs
potim    = 1.0

THzToCm = 33.3564095198152
omega   = THzToCm * 1E3 * fftfreq(nsw, potim)
psd     = np.abs(fft(et))**2
np.savetxt('ft.dat', np.c_[omega[:nsw//2], psd[:nsw//2]])

# plotting
import matplotlib.pyplot as plt
fig = plt.figure()
ax  = plt.subplot()

ax.plot(omega[:nsw//2], psd[:nsw//2], 'r-')

ax.set_xlim(0, 2000)

ax.set_xlabel(r'Wavenumber [cm$^{-1}$]', labelpad=5)
ax.set_ylabel('Amplitude', labelpad=5)

plt.show()
