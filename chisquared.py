from kilonova_heating_rate import lightcurve
from synphot import SpectralElement

import astropy.units as u
import astropy.constants as c
import numpy as np
from sed_integrator import get_abmag
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import dorado.sensitivity

def calc_bolometric(t,params):
    v_k, v_max, kappa_low = params
    mass = 0.05 * u.Msun
    velocities = np.asarray([0.1, v_k, v_max]) * c.c
    opacities = np.asarray([3.0, kappa_low]) * u.cm**2 / u.g
    n = 4.5
    heating = 'beta'
    
    L, T, r = lightcurve(t, mass, velocities, opacities, n, heating_function=heating)
    return L, T, r

def chi_squared_UVOT(params):
    print(params)
    labels = ['UVW1', 'UVM2', 'UVW2']
    bandpasses = [SpectralElement.from_file(f'input_files/Swift_UVOT.{label}.dat') for label in labels]
    t = np.geomspace(0.02, 2) * u.day
    distance = 10 * u.pc
    L, T, r = calc_bolometric(t,params)
    abmags = [get_abmag(T,r,distance,bandpass) for bandpass in bandpasses]
    interps = [interp1d(t, abmag) for abmag in abmags]

    # Calculate Chi squared parameter for datapoints
    datapoints = [np.asarray([[0.6, 1],[-14.2,-13.5]]),np.asarray([[0.6,1],[-12.6,-11.3]]),np.asarray([0.6,-12.6])] #banerjee 2020
    chisquared_datapoints = [(interp(datapoint[0])-datapoint[1])**2 for interp, datapoint in zip(interps, datapoints)]
    
    # Calculate Chi Squared parameter for peak values
    peaks_data = [-16, -15.9, -15.6] #banerjee 2020
    peaks_model = [min(abmag) for abmag in abmags]
    chisquared_peaks = [(peak_data-peak_model)**2 for peak_data,peak_model in zip(peaks_data,peaks_model)]
    
    
    ## Sum Chi squared parameters
    chisquared_total = []
    for chisquared_datapoint, chisquared_peak in zip(chisquared_datapoints, chisquared_peaks):
        try:
            chisquared_summed = sum(chisquared_datapoint)
        except:
            chisquared_summed = chisquared_datapoint
        
        chisquared_total.append(chisquared_peak+chisquared_summed)
    result = sum(chisquared_total)
    print(result)
    return result


v_k = 0.2
v_max = 0.3
kappa_low = 0.5
initial_guess = [v_k, v_max, kappa_low]
bounds = [(0.1,0.2),(0.2,0.8),(0.0005,3)]
result = optimize.minimize(chi_squared_UVOT, initial_guess, bounds=bounds)
if result.success:
    v_k_optimized, v_max_optimized, kappa_low_optimized = result.x
    print(f'v_k: {v_k_optimized}')
    print(f'v_max: {v_max_optimized}')
    print(f'kappa_low: {kappa_low_optimized}')
else:
    raise ValueError(result.message)
    
labels = ['UVW1', 'UVM2', 'UVW2']
t = np.geomspace(0.02, 2) * u.day
distance = 10 * u.pc
datapoints = [np.asarray([[0.6, 1],[-14.2,-13.5]]),np.asarray([[0.6,1],[-12.6,-11.3]]),np.asarray([0.6,-12.6])] #banerjee 2020
L, T, r = calc_bolometric(t,[v_k_optimized, v_max_optimized, kappa_low_optimized])
L_hotokezaka, T_hotokezaka, r_hotokezaka = calc_bolometric(t,[0.2,0.4,0.5])
bandpasses = [SpectralElement.from_file(f'input_files/Swift_UVOT.{label}.dat') for label in labels]
abmags = [get_abmag(T,r,distance,bandpass) for bandpass in bandpasses]

bp_dorado = dorado.sensitivity.bandpasses.NUV_D
abmag_dorado = get_abmag(T,r,distance,bp_dorado)
abmag_dorado_hotokezaka = get_abmag(T_hotokezaka,r_hotokezaka,distance,bp_dorado)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1,figsize=(6,10))
fig.suptitle(f'v_k_opt: {v_k_optimized:.2f}, v_max_opt: {v_max_optimized:.2f}, k_low_opt: {kappa_low_optimized:.2f}')
ax[0].invert_yaxis()
ax[1].invert_yaxis()

# for interp, datapoint,label in zip(interps, datapoints, labels):
#     ax.plot(datapoint[0],interp(datapoint[0]),'x',label=f'{label} model')
#     ax.plot(datapoint[0],datapoint[1],'o',label=f'{label} data')

for label, abmag, datapoint in zip(labels, abmags, datapoints):
    ax[0].plot(datapoint[0],datapoint[1],'x',label=label)
    ax[0].plot(t,abmag,label=label)

ax[0].legend()
ax[0].set_ylabel('AB mag (abs')
ax[1].plot(t,abmag_dorado,label='optimized')
ax[1].plot(t,abmag_dorado_hotokezaka,label='hotokezaka')
ax[1].set_ylabel('AB mag (abs)')
ax[1].legend()
ax[2].semilogy(t,L,label='optimized')
ax[2].semilogy(t,L_hotokezaka,label='hotokezaka')
ax[2].legend()
ax[2].set_ylabel('bol lum (erg/s)')
ax[2].set_xlabel('time (days)')
plt.tight_layout()
fig.savefig('output_files/plot.png')