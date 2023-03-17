import matplotlib.pyplot as plt
import numpy as np
from consts import *
from uncertainties import ufloat

def norm_hist(array, bins, range_start,range_end, norm,scale=1):
    # make normalized histogram
    y,binEdges = np.histogram(np.array(array),bins=bins,range=(range_start,range_end))
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    menStd     = np.sqrt(y)
    y_norm = np.zeros_like(y,dtype='float64')
    y_norm_std = np.zeros_like(y,dtype='float64')
    
    bincontents_total = np.sum(y)
    for i in range(len(y)):
        y_uncert = ufloat(y[i],menStd[i])
        if norm == 'area':
            y_uncert = y_uncert/bincontents_total
        elif norm == 'max':
            y_uncert = y_uncert/np.max(y)
        else:
            y_uncert = y_uncert
        y_uncert *= scale
        y_norm[i] = y_uncert.nominal_value
        y_norm_std[i] = y_uncert.std_dev
    return bincenters, y_norm, y_norm_std

def plot_hist(data,plots,bins,data_type,color='b',label=None, calibrate=False,norm='none',scale=1,recomb_filename=None):
    linewidth = 1.5
    vcm_mv = v_cm_data
    vref_mv = v_ref_data
    gain = gain_data

    if data_type == 'MC':
        vcm_mv = v_cm_sim
        vref_mv = v_ref_sim
        gain = gain_sim
    LSB = (vref_mv - vcm_mv)/256
    width = LSB / gain * 1e-3
    
    offset = np.zeros_like(data)
    if data_type == 'MC':
        MC_size = len(data)
        offset = scipy.stats.uniform.rvs(loc=0, scale=0.5, size=MC_size)*width*np.random.choice([-1, 1], size=MC_size, p=[.5, .5])
        data += offset

    range_start = width*0 # ke-
    range_end = width*bins
    eV_per_e = 1
    if calibrate:
        eV_per_e = 23.6
        if recomb_filename == None:
            raise Exception("Calibrate is set to True, so must provide non-Null filename of h5 file with energies and recombination values.")
        else:
            recomb_file = h5py.File(recomb_filename)
            energies = np.array(recomb_file['NEST']['E_start'])
            recombination = np.array(recomb_file['NEST']['R'])
            charge_ke = energies / 23.6 * recombination
            data = data/np.interp(data, charge_ke, recombination)


    bincenters, y_norm, y_norm_std = norm_hist(data*eV_per_e, bins, range_start*eV_per_e,range_end*eV_per_e,norm,scale)
    #axes.bar(bincenters_data, y_norm_data, width=width, color=color, yerr=y_norm_std_data,alpha=0.2, label=label)
    plots.step(bincenters, y_norm, linewidth=linewidth, color=color,where='mid',alpha=0.7, label=label)
    plots.errorbar(bincenters, y_norm, yerr=y_norm_std,color='k',fmt='o',markersize = 1)
    if calibrate:
        plots.set_xlabel('keV')
    else:
        plots.set_xlabel('ke-')
    return bincenters, y_norm, y_norm_std
