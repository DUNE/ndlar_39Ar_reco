#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

def XY_Hist2D(clusters, figTitle=None, vmin=1e0, vmax=1e3):
    ### plot 2D histogram of clusters
    y_min_max = [-620,620]
    x_min_max = [-310,310]
    x_bins = 140
    y_bins = 2*x_bins
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8,6))
    cmap = plt.cm.jet
    z_anode_max = np.max(clusters['z_anode'])
    z_anode_min = np.min(clusters['z_anode'])
    
    TPC1_mask = (clusters['z_anode'] < 0) & (clusters['z_drift_mid'] > z_anode_min) & (clusters['z_drift_mid'] < 0)
    TPC2_mask = (clusters['z_anode'] > 0) & (clusters['z_drift_mid'] < z_anode_max) & (clusters['z_drift_mid'] > 0)
    
    H1 = axes[0].hist2d(clusters['x_mid'][TPC1_mask], clusters['y_mid'][TPC1_mask], range=[x_min_max, y_min_max],bins = [x_bins,y_bins], weights=np.ones_like(clusters['x_mid'][TPC1_mask]),norm = colors.LogNorm(vmin=vmin,vmax=vmax))
    fig.colorbar(H1[3], ax=axes[0])
    H2 = axes[1].hist2d(clusters['x_mid'][TPC2_mask], clusters['y_mid'][TPC2_mask], range=[x_min_max, y_min_max], bins = [x_bins,y_bins], weights=np.ones_like(clusters['x_mid'][TPC2_mask]),norm = colors.LogNorm(vmin=vmin,vmax=vmax))
    fig.colorbar(H2[3], ax=axes[1])
    axes[0].set_title(f'TPC 1')
    axes[1].set_title(f'TPC 2')
    fig.suptitle(figTitle, fontsize=10)
    axes[0].set_xlabel(r'$x_{reco}$ [mm]')
    axes[1].set_xlabel(r'$x_{reco}$ [mm]')
    axes[0].set_ylabel(r'$y_{reco}$ [mm]')
    axes[0].set_ylim(y_min_max[0], y_min_max[1])
    axes[0].set_xlim(x_min_max[0], x_min_max[1])
    axes[1].set_ylim(y_min_max[0], y_min_max[1])
    axes[1].set_xlim(x_min_max[0], x_min_max[1])
    plt.show()

def XZ_Hist2D(clusters, figTitle=None, logYscale=False, vmin=1, vmax=1e3):
    ### plot 2D histogram of clusters
    x_min_max = [-310,310]
    x_bins = 140
    y_bins = x_bins
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8,6))
    cmap = plt.cm.jet
    
    z_anode_max = np.max(clusters['z_anode'])
    z_anode_min = np.min(clusters['z_anode'])
    
    if logYscale:
        norm = colors.LogNorm(vmin=1e0,vmax=1e+3)
    else:
        norm = None
    
    TPC1_mask = (clusters['z_anode'] < 0) & (clusters['z_drift_mid'] > z_anode_min) & (clusters['z_drift_mid'] < 0)
    TPC2_mask = (clusters['z_anode'] > 0) & (clusters['z_drift_mid'] < z_anode_max) & (clusters['z_drift_mid'] > 0)
    clusters_TPC1 = clusters[TPC1_mask]
    clusters_TPC2 = clusters[TPC2_mask]
    clusters_fiducial = np.concatenate((clusters_TPC1, clusters_TPC2))
    
    H1 = axes.hist2d(clusters_fiducial['x_mid'], clusters_fiducial['z_drift_mid'], range=[x_min_max, x_min_max],bins = [x_bins,y_bins], weights=np.ones_like(clusters_fiducial['x_mid']), vmin=vmin, vmax=vmax,norm = norm)
    axes.set_xlabel(r'$x_{reco}$ [mm]')
    axes.set_ylabel(r'$z_{reco}$ [mm]')
    axes.set_ylim(x_min_max[0], x_min_max[1])
    axes.set_xlim(x_min_max[0], x_min_max[1])
    fig.suptitle(figTitle)

    plt.show()

def make_hist(array, bins, range_start, range_end):
    ### make histogram of charge
    
    # get histogram data
    bin_contents,binEdges = np.histogram(np.array(array),bins=bins,range=(range_start,range_end))
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    error      = np.sqrt(bin_contents)
    
    return bincenters, bin_contents, error

def get_hist_data(clusters, bins, data_type, calibrate=False, binwidth=None, recomb_filename=None):
    ### set bin size and histogram range, correct for recombination using NEST, return histogram parameters
    # INPUT: `the_data` is a 1D numpy array of charge cluster charge values (mV)
    #        `bins` is the number of bins to use (this effectively sets the range, the binsize is constant w.r.t. bins)
    #        `data_type` is either `data` for real data or `MC` for simulation
    #        `calibrate` is either True or False, if True will use NEST to correct for recombination
    #        `norm` sets the normalization of histogram. Options are `area` and `max`, otherwise `None`
    #        `binwidth` is optional to specify binwidth. Otherwise will be 2*LSB by default
    #        `recomb_filename` is path to h5 file containing electron recombination values as a function of energy
    
    v_cm_sim = 288.28125
    v_ref_sim = 1300.78125
    v_pedestal_sim = 580
    
    v_cm_data = 288.28125
    v_ref_data = 1300.78125
    
    gain_data = 1/221
    gain_sim = 1/221
    data = np.copy(clusters['q'])
    
    # set parameters for data or MC for determining bin size
    vcm_mv = v_cm_data
    vref_mv = v_ref_data
    gain = gain_data
    if calibrate:
        data = data/gain * 1e-3
    
    if data_type == 'MC':
        vcm_mv = v_cm_sim
        vref_mv = v_ref_sim
        gain = gain_sim
    LSB = (vref_mv - vcm_mv)/256
    if calibrate:
        width = LSB / gain * 1e-3
    else:
        width = LSB
    
    # add small +/- offset in MC for fix binning issues
    offset = np.zeros_like(data)
    if data_type == 'MC':
        MC_size = len(data)
        offset = scipy.stats.uniform.rvs(loc=0, scale=1, size=MC_size)*width*np.random.choice([-0.5,0.5], size=MC_size, p=[.5, .5])
        data += offset
    if binwidth is not None:
        width = binwidth
    
    range_start = width*0 # ke-
    range_end = width*bins
    
    # if converting to energy, use NEST model
    eV_per_e = 1
    if recomb_filename is not None:
        eV_per_e = 23.6
        recomb_file = h5py.File(recomb_filename)
        energies = np.array(recomb_file['NEST']['E_start'])
        recombination = np.array(recomb_file['NEST']['R'])
        charge_ke = energies / 23.6 * recombination
        data = data/np.interp(data, charge_ke, recombination)
    
    # get histogram parameters
    nbins = int(bins/2)
    bin_centers, bin_contents, bin_error = make_hist(data*eV_per_e, nbins, range_start*eV_per_e,range_end*eV_per_e)
    return bin_centers, bin_contents, bin_error

def plotRecoSpectrum(clusters, nbins=100, data_type='data', color='b', linewidth=1, label=None, linestyle=None, norm=None,plot_errorbars=False, useYlog=False, calibrate=True):
    ### plot reco spectrum
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6,4))
    bin_centers, bin_contents, bin_error = get_hist_data(clusters, nbins, data_type, calibrate=calibrate)
    if norm == 'area':
        total_bin_contents = np.sum(bin_contents)
        bin_contents = bin_contents / total_bin_contents
        bin_error = bin_error / total_bin_contents
        axes.set_ylabel('bin count / total bin count')
    elif norm == 'max':
        max_bin_content = np.max(bin_contents)
        bin_contents = bin_contents / max_bin_content
        bin_error = bin_error / max_bin_content
        axes.set_ylabel('bin count / max bin count')
    else:
        axes.set_ylabel('bin count')
    if calibrate:
        axes.set_xlabel('Charge [ke-]')
    else:
        axes.set_xlabel('Charge [mV]')
    axes.step(bin_centers, bin_contents, linewidth=linewidth, color=color,linestyle=linestyle, where='mid',alpha=0.7, label=label)
    if useYlog:
        axes.set_yscale('log')
    if plot_errorbars:
        axes.errorbar(bin_centers, bin_contents, yerr=bin_error,color='k',fmt='o',markersize = 1)
    
def get_charge_MC(nFiles_dict, folders_MC, filename_ending_MC, nbins, do_calibration, recomb_filename,disable_alphas=False, disable_gammas=False, disable_betas=False):
    # Isotope ratios
    isotopes_ratios_betas_gammas = { 
        '85Kr': 224.76, # beta/gamma ratio
        '60Co': 0.5,
        '40K': 8.46
    }
    
    isotopes_ratios_betas_alphas_gammas_alphas = {
        '232Th': [0.649, 0.45], # betas/alphas , gammas/alphas ratios
        '238U': [0.751, 0.999]
    }
    
    # Initialize dictionaries
    charge_dict = {}
    hist_data_dict = {}
    
    # Loop over isotopes
    for iso_decay, nFiles in nFiles_dict.items():
        iso, decay = iso_decay.split('_')
        folder = folders_MC[iso_decay]
        ending = filename_ending_MC[iso_decay]
        # Loop over files
        for i in range(1, nFiles+1):
            f = h5py.File(folder + f'larndsim_{iso}_{decay}_10000_{i}_{ending}.h5', 'r')
            charge_temp = f['clusters']['q']
            if i == 1:
                charge_dict[iso_decay] = charge_temp
            else:
                charge_dict[iso_decay] = np.concatenate((charge_dict[iso_decay], charge_temp))
        
        # Call function to get histogram data
        bin_centers, bin_contents, bin_error = \
            get_hist_data(charge_dict[iso_decay], bins=nbins, data_type='MC', \
            calibrate=do_calibration, recomb_filename=recomb_filename)
        
        hist_data_dict[iso_decay] = {
            'bin_centers': bin_centers,
            'bin_contents': bin_contents,
            'bin_error': bin_error
        }
    
    # Combine y_norm and y_norm_std for isotopes that have betas and gammas
    for iso in isotopes_ratios_betas_gammas.keys():
        R = isotopes_ratios_betas_gammas[iso]
        x_1 = R * (np.sum(hist_data_dict[iso+'_gammas']['bin_contents']) / np.sum(hist_data_dict[iso+'_betas']['bin_contents']))
        x_2 = 1
        
        if disable_gammas:
            x_2 = 0
        if disable_betas:
            x_1 = 0
        hist_data_dict[iso] = {
            'bin_centers': hist_data_dict[iso+'_betas']['bin_centers'],
            'bin_contents': hist_data_dict[iso+'_betas']['bin_contents']*x_1 + hist_data_dict[iso+'_gammas']['bin_contents']*x_2,
            'bin_error': np.sqrt((hist_data_dict[iso+'_betas']['bin_error']*x_1)**2 + (hist_data_dict[iso+'_gammas']['bin_error']*x_2)**2)
        }
    for iso in isotopes_ratios_betas_alphas_gammas_alphas.keys():
        R = isotopes_ratios_betas_alphas_gammas_alphas[iso]
        x_1 = R[0] * (np.sum(hist_data_dict[iso+'_alphas']['bin_contents']) / np.sum(hist_data_dict[iso+'_betas']['bin_contents']))
        x_2 = R[1] * (np.sum(hist_data_dict[iso+'_alphas']['bin_contents']) / np.sum(hist_data_dict[iso+'_gammas']['bin_contents']))
        x_3 = 1
        if disable_betas:
            x_1 = 0
        if disable_gammas:
            x_2 = 0
        if disable_alphas:
            x_3 = 0
        hist_data_dict[iso] = {
            'bin_centers': hist_data_dict[iso+'_betas']['bin_centers'],
            'bin_contents': hist_data_dict[iso+'_betas']['bin_contents']*x_1 + hist_data_dict[iso+'_gammas']['bin_contents']*x_2 + \
                (hist_data_dict[iso+'_alphas']['bin_contents']*x_3),
            'bin_error': np.sqrt((hist_data_dict[iso+'_betas']['bin_error']*x_1)**2 + (hist_data_dict[iso+'_gammas']['bin_error']*x_2)**2 \
                                + (hist_data_dict[iso+'_alphas']['bin_error']*x_3)**2)
        }
    return charge_dict, hist_data_dict

def plot_isotopes(hist_data_dict, axes, colors, norm=None, linewidth=2, do_not_plot_list=None):    
    # Loop over isotopes
    for iso_decay, color in colors.items():
        # Get histogram data
        bin_centers = hist_data_dict[iso_decay]['bin_centers']
        bin_contents = hist_data_dict[iso_decay]['bin_contents']
        bin_error = hist_data_dict[iso_decay]['bin_error']
        
        if len(iso_decay.split('_')) > 1:
            label = iso_decay.split('_')[0]
        else:
            label = iso_decay
        if label not in do_not_plot_list:
            # Call function to plot histogram
            plot_hist(bin_centers, bin_contents, bin_error, axes, color, linewidth, label, norm=norm)
    

