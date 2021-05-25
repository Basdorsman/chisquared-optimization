from sed_integrator import get_abmag
from kilonova_heating_rate import lightcurve
import astropy.units as u
import astropy.constants as c
import numpy as np
from synphot import SpectralElement



def calc_bolometric(t,params):
    v_k, v_max, kappa_low = params
    mass = 0.05 * u.Msun
    velocities = np.asarray([0.1, v_k, v_max]) * c.c
    opacities = np.asarray([3.0, kappa_low]) * u.cm**2 / u.g
    n = 4.5
    heating = 'beta'
    
    L, T, r = lightcurve(t, mass, velocities, opacities, n, heating_function=heating)
    return L, T, r

data_banerjee = np.loadtxt('input_files/mag.dat')
UVW1 = data_banerjee[:,17]
UVW2 = data_banerjee[:,18]
UVM2 = data_banerjee[:,19]
banerjee_UVs = [UVW1, UVM2, UVW2]
t_banerjee = data_banerjee[:,0]


v_k = 0.2
v_max = 0.23
kappa_low = 0.04
params = [v_k, v_max, kappa_low]
labels = ['UVW1', 'UVM2', 'UVW2']
colors = ['skyblue','mediumvioletred','navy']
bandpasses = [SpectralElement.from_file(f'input_files/Swift_UVOT.{label}.dat') for label in labels]
t = np.geomspace(0.02, 2) * u.day
distance = 10 * u.pc
L, T, r = calc_bolometric(t,params)
abmags = [get_abmag(T,r,distance,bandpass) for bandpass in bandpasses]
datapoints = [np.asarray([[0.6, 1],[-14.2,-13.5]]),np.asarray([[0.6,1],[-12.6,-11.3]]),np.asarray([0.6,-12.6])] #banerjee 2020

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for banerjee_UV, label, abmag, datapoint, color in zip(banerjee_UVs, labels, abmags, datapoints, colors):
    ax.semilogx(t_banerjee, banerjee_UV,'--',label=f'{label} banerjee',color=color)
    ax.semilogx(datapoint[0],datapoint[1],'x',label=f'{label} swift data',color=color)
    ax.semilogx(t,abmag,label=f'{label} lower early opacity',color=color)

ax.invert_yaxis()
ax.set_ylim([-10,-16.5])
ax.set_xlim([0.01,2])
ax.set_xlabel('time (days)')
ax.set_ylabel('AB magnitude (apparent)')
ax.grid(1)
ax.legend()
fig.savefig('output_files/plot.png')