#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.fftpack import fftfreq, fft, ifft

def gaussian(x,c):
    return np.exp(-x**2/(2*c**2))

def dephase(Et, dt=1.0):
    '''
    Calculate the autocorrelation function (ACF), dephasing function, and FT of
    ACF.

    The dephasing function was calculated according to the following formula

    G(t) = (1 / hbar**2) \int_0^{t_1} dt_1 \int_0^{t_2} dt_2 <E(t_2)E(0)>
    D(t) = exp(-G(t))

    where Et is the difference of two KS energies in unit of eV, <...> is the
    ACF of the energy difference and the brackets denote canonical averaging.

    Fourier Transform (FT) of the normalized ACF gives the phonon influence
    spectrum, also known as the phonon spectral density.

    I(\omega) \propto | FT(Ct / Ct[0]) |**2

    Jaeger, Heather M., Sean Fischer, and Oleg V. Prezhdo. "Decoherence-induced surface hopping." JCP 137.22 (2012): 22A545.
    '''

    from scipy.integrate import cumtrapz
    from scipy.fftpack import fft

    hbar = 0.6582119513926019       # eV fs

    Et = np.asarray(Et)
    Et -= np.average(Et)

    # Autocorrelation Function (ACF) of Et
    Ct = np.correlate(Et, Et, 'full')[Et.size:] / Et.size
    
    # Cumulative integration of the ACF
    Gt = cumtrapz(cumtrapz(Ct, dx=dt, initial=0), dx=dt, initial=0)
    Gt /= hbar**2
    # Dephasing function
    Dt = np.exp(-Gt)

    # FT of normalized ACF
    Iw = np.abs(fft(Ct / Ct[0]))**2

    return Ct, Dt, Iw

dt = 1.0 # fs
T = np.arange(1999) * dt
nsw      = 2000
potim    = 1.0
THzToCm = 33.3564095198152
omega   = THzToCm * 1E3 * fftfreq(nsw, potim)

#####################this area  must be changed in the script########################## 
energy_pristine = np.loadtxt('/fs1/home/sduniversity/bly/Tm_LaCOB/LaCOB_dielectric/NVT_v_s/NVE/dish/energy.dat')
Et_pristine = energy_pristine[:, 1] - energy_pristine[:, 0]
Ct_pristine, Dt_pristine, Iw_pristine = dephase(Et_pristine)
Iw_pristine = Iw_pristine/np.linalg.norm(Iw_pristine)

energy_TmCa0 = np.loadtxt('/fs1/home/sduniversity/bly/Tm_LaCOB/TmCa2/TmCa0/NVT_v_s1/NVE/dish/energy.dat')
Et_TmCa0 = energy_TmCa0[:, 2] - energy_TmCa0[:, 0]
Ct_TmCa0, Dt_TmCa0, Iw_TmCa0 = dephase(Et_TmCa0)
Iw_TmCa0 = Iw_TmCa0/np.linalg.norm(Iw_TmCa0)
#######################################################################################

import matplotlib.pyplot as plt
FS = 18
plt.rcParams['font.size'] = FS
plt.rcParams['axes.labelsize'] = FS
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS
plt.rcParams['font.sans-serif'] = ['Arial']

fig,ax = plt.subplots(3,1,figsize=(8,15),dpi=600)
ax[0].set_title('(a)',loc='left', weight='regular')
ax[1].set_title('(b)',loc='left', weight='regular')
ax[2].set_title('(c)',loc='left', weight='regular')
x = np.array([i for i in range(0,1998)])
ax[0].plot(x,Ct_pristine,label='Pristine')
ax[0].plot(x,Ct_TmCa0,label='Tm$_C$$_a$$^0$')
ax[0].set_xlim(-8,2000)
ax[0].set_xticks([0,500,1000,1500,2000])
ax[0].set_xlabel('Wavenumber [Time [fs]')
ax[0].set_ylabel('C$_u$$_n$[t] [eV$^2$]')

ax[1].plot(x,Dt_pristine,label='Pristine')
ax[1].plot(x,Dt_TmCa0,label='Tm$_C$$_a$$^0$')
ax[1].set_xlim(0,15)
ax[1].set_ylim(0,1)
ax[1].set_xlabel('Time [fs]')
ax[1].set_ylabel('Dephasing')

ax[2].plot(omega[:1998],Iw_pristine,label='Pristine')
ax[2].plot(omega[:1998],Iw_TmCa0,label='Tm$_C$$_a$$^0$')
ax[2].set_xlim(0,1000)
ax[2].set_ylim(0,0.5)
ax[2].set_yticks([])
ax[2].set_xlabel('Wavenumber [cm$^-1$]')
ax[2].set_ylabel('Spectral Density')
for i in ax:
    i.legend(loc='upper right')
plt.savefig('ACF,Dt,FFT.png')

