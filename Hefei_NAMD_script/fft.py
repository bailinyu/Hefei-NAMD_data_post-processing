#!/usr/bin/env python

import numpy as np
from scipy.fftpack import fftfreq, fft, ifft

# Two-column data
et       = np.loadtxt('ENRTXT')[:,0]
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
