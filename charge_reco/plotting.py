#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy import stats
import matplotlib.colors as mcolors

def XY_Hist2D(clusters, figTitle=None, vmin=1e0, vmax=1e3, use_z_cut=True, isSingleCube=False):
    ### plot 2D histogram of clusters
    if isSingleCube:
        y_min_max = [-155,155]
        x_min_max = [-155,155]
        x_bins = 70
        y_bins = x_bins
        ncols=1
    else:
        y_min_max = [-620,620]
        x_min_max = [-310,310]
        x_bins = 140
        y_bins = 2*x_bins
        ncols=2
    fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=False, sharey=False, figsize=(8,6))
    cmap = plt.cm.jet
    z_anode_max = np.max(clusters['z_anode'])
    z_anode_min = np.min(clusters['z_anode'])
    if isSingleCube:
        H1 = axes.hist2d(clusters['x_mid'], clusters['y_mid'], range=[x_min_max, y_min_max],bins = [x_bins,y_bins], weights=np.ones_like(clusters['x_mid']),norm = colors.LogNorm(vmin=vmin,vmax=vmax))
        fig.colorbar(H1[3], ax=axes)
        
        #axes[0].set_title(f'TPC 1')
        fig.suptitle(figTitle, fontsize=10)
        axes.set_xlabel(r'$x_{reco}$ [mm]')
        axes.set_ylabel(r'$y_{reco}$ [mm]')
        axes.set_ylim(y_min_max[0], y_min_max[1])
        axes.set_xlim(x_min_max[0], x_min_max[1])
    else:
        if not use_z_cut:
            TPC1_mask = (clusters['z_anode'] < 0)
            TPC2_mask = (clusters['z_anode'] > 0)
        else:
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

def XZ_Hist2D(clusters, figTitle=None, logYscale=False, vmin=1, vmax=1e3, weight_type=None):
    ### plot 2D histogram of clusters
    x_min_max = [-310,310]
    x_bins = 140
    y_bins = x_bins
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8,6))
    cmap = plt.cm.jet
    
    z_anode_max = np.max(clusters['z_anode'])
    z_anode_min = np.min(clusters['z_anode'])
    
    if logYscale:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = None
    
    TPC1_mask = (clusters['z_anode'] < 0) & (clusters['z_drift_mid'] > z_anode_min) & (clusters['z_drift_mid'] < 0)
    TPC2_mask = (clusters['z_anode'] > 0) & (clusters['z_drift_mid'] < z_anode_max) & (clusters['z_drift_mid'] > 0)
    clusters_TPC1 = clusters[TPC1_mask]
    clusters_TPC2 = clusters[TPC2_mask]
    clusters_fiducial = np.concatenate((clusters_TPC1, clusters_TPC2))
    if weight_type == 'q':
        bin_counts = np.histogram2d(clusters_tagged['x_mid'], clusters_tagged['y_mid'], bins=[x_bins,y_bins], range=[x_min_max,x_min_max])[0]
        weights = clusters_fiducial['q']*221*1e-3
    else:
        weights = np.ones_like(clusters_fiducial['x_mid'])
    H1 = axes.hist2d(clusters_fiducial['x_mid'], clusters_fiducial['z_drift_mid'], \
                     range=[x_min_max,x_min_max],bins = [x_bins,y_bins], \
                     weights=weights, vmin=vmin, vmax=vmax,norm = norm)
    axes.set_xlabel(r'$x_{reco}$ [mm]')
    axes.set_ylabel(r'$z_{reco}$ [mm]')
    axes.set_ylim(x_min_max[0], x_min_max[1])
    axes.set_xlim(x_min_max[0], x_min_max[1])
    fig.suptitle(figTitle)

    plt.show()

def plot_2D_statistic(clusters, values, stat, plot_type, xlabel=None, ylabel=None, figTitle=None, vmin=None, vmax=None, log_scale=False, isSingleCube=False):
    if plot_type == 'xy':
        if isSingleCube:
            ncols=1
            figsize=(8,6)
        else:
            ncols=2
            figsize=(8,6)
    elif plot_type == 'xz':
        ncols=1
        figsize=(8,6)
    else:
        raise Exception('plot type not supported')
    fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=False, sharey=False, figsize=figsize)
    cmap = plt.cm.jet
        
    if plot_type == 'xz':
        RANGE = [-310,310]
        x_bins = 140
        y_bins = x_bins
        axes.set_xlabel(r'$x_{reco}$ [mm]')
        axes.set_ylabel(r'$z_{reco}$ [mm]')
        axes.set_ylim(RANGE[0], RANGE[1])
        axes.set_xlim(RANGE[0], RANGE[1])
    elif plot_type == 'xy':
        if isSingleCube:
            y_min_max = [-155,155]
            x_min_max = [-155,155]
            RANGE=[x_min_max, y_min_max]
            x_bins = 70
            y_bins = x_bins
            axes.set_xlabel(r'$x_{reco}$ [mm]')
            axes.set_ylabel(r'$y_{reco}$ [mm]')
            axes.set_ylim(y_min_max[0], y_min_max[1])
            axes.set_xlim(x_min_max[0], x_min_max[1])
        else:
            y_min_max = [-620,620]
            x_min_max = [-310,310]
            RANGE=[x_min_max, y_min_max]
            x_bins = 140
            y_bins = 2*x_bins
            axes[0].set_xlabel(r'$x_{reco}$ [mm]')
            axes[1].set_xlabel(r'$x_{reco}$ [mm]')
            axes[0].set_ylabel(r'$y_{reco}$ [mm]')
            axes[0].set_ylim(y_min_max[0], y_min_max[1])
            axes[0].set_xlim(x_min_max[0], x_min_max[1])
            axes[1].set_ylim(y_min_max[0], y_min_max[1])
            axes[1].set_xlim(x_min_max[0], x_min_max[1])
    if plot_type == 'xy':
        hist_data = stats.binned_statistic_2d(clusters['x_mid'], clusters['y_mid'], values, statistic=stat, bins=[x_bins,y_bins], range=RANGE)
    elif plot_type == 'xz':
        hist_data = stats.binned_statistic_2d(clusters['x_mid'], clusters['z_drift_mid'], values, statistic=stat, bins=[x_bins,y_bins], range=[RANGE, RANGE])

    if log_scale:
        norm = mcolors.LogNorm(vmin=np.nanmin(hist_data.statistic), vmax=np.nanmax(hist_data.statistic))
    else:
        norm = None
    if plot_type == 'xz':
        im = axes.imshow(hist_data.statistic.T, cmap=cmap,norm=norm, origin='lower', extent=[RANGE[0], RANGE[1], RANGE[0], RANGE[1]])
        colorbar = plt.colorbar(im, ax=axes)
        im.set_clim(vmin, vmax)
    elif plot_type == 'xy':
        if isSingleCube:
            im = axes.imshow(hist_data.statistic.T, cmap=cmap,norm=norm, origin='lower', extent=[RANGE[0][0], RANGE[0][1], RANGE[1][0], RANGE[1][1]])
            im.set_clim(vmin, vmax)
        else:
            im = axes[0].imshow(hist_data.statistic.T, cmap=cmap,norm=norm, origin='lower', extent=[RANGE[0][0], RANGE[0][1], RANGE[1][0], RANGE[1][1]])
            im.set_clim(vmin, vmax)
            im = axes[1].imshow(hist_data.statistic.T, cmap=cmap, norm=norm, origin='lower', extent=[RANGE[0][0], RANGE[0][1], RANGE[1][0], RANGE[1][1]])
        colorbar = plt.colorbar(im, ax=axes)
        im.set_clim(vmin, vmax)
    fig.suptitle(figTitle)
    

def make_hist(array, bins, range_start, range_end):
    ### make histogram of charge
    
    # get histogram data
    bin_contents,binEdges = np.histogram(np.array(array),bins=bins,range=(range_start,range_end))
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    error      = np.sqrt(bin_contents)
    
    return bincenters, bin_contents, error

def get_hist_data(clusters, bins, data_type, calibrate=False, binwidth=None, recomb_filename=None, bin_start=0):
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
    if DET == 'module-0':
        v_cm_data = 288.28125
        v_ref_data = 1300.78125
    elif DET == 'module-1':
        v_cm_data = 284.27734375
        v_ref_data = 1282.71484375
    else:
        v_cm_data = 288.28125
        v_ref_data = 1300.78125 
    # module-2
    #v_cm_data = 438.28125
    #v_ref_data = 1437.3046875
    
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
    if data_type == 'MC' or data_type == 'data':
        MC_size = len(data)
        offset = stats.uniform.rvs(loc=0, scale=1, size=MC_size)*width*np.random.choice([-0.5,0.5], size=MC_size, p=[.5, .5])
        data += offset
    if binwidth is not None:
        width = binwidth
    
    range_start = width*bin_start # ke-
    range_end = width*(bins+bin_start)
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
    nbins = int(bins)
    bin_centers, bin_contents, bin_error = make_hist(data*eV_per_e, nbins, range_start*eV_per_e,range_end*eV_per_e)
    return bin_centers, bin_contents, bin_error
def plotRecoSpectrum(clusters, nbins=100, data_type='data', color='b', linewidth=1, label=None, linestyle=None, norm=None,plot_errorbars=False, useYlog=False
, calibrate=True, bin_start=0, axes=None, recomb_filename=None, DET=None, figTitle=None, saveFig=False, fileName=None)
    ### plot reco spectrum
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6,4))
    bin_centers, bin_contents, bin_error = get_hist_data(clusters, nbins, data_type, calibrate=calibrate, bin_start=bin_start, recomb_filename=recomb_filenam
e, DET=DET)
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
    if calibrate and recomb_filename is None:
        axes.set_xlabel('Charge [ke-]')
    elif not calibrate and recomb_filename is None:
        axes.set_xlabel('Charge [mV]')
    elif calibrate and recomb_filename is not None:
        axes.set_xlabel('Cluster Energy [keV]')
    axes.step(bin_centers, bin_contents, linewidth=linewidth, color=color,linestyle=linestyle, where='mid',alpha=0.7, label=label)
    if useYlog:
        axes.set_yscale('log')
    if plot_errorbars:
        axes.errorbar(bin_centers, bin_contents, yerr=bin_error,color='k',fmt='o',markersize = 1)

def linear_fit(x, y, error, axes, make_plot=True):
    # Weighted least squares regression
    error[error == 0] = 1.15
    weights = 1 / (error ** 2)
    # Calculate slope (m) and intercept (b)
    sum_w = np.sum(weights)
    sum_wx = np.sum(weights * x)
    sum_wy = np.sum(weights * y)
    sum_wxx = np.sum(weights * x ** 2)
    sum_wxy = np.sum(weights * x * y)
    print(f'sum_w={sum_w}; sum_wx={sum_wx}; sum_wy={sum_wy}; sum_wxx={sum_wxx}; sum_wxy={sum_wxy}')
    m = (sum_wxy - (sum_wx * sum_wy) / sum_w) / (sum_wxx - (sum_wx ** 2) / sum_w)
    b = (sum_wy - m * sum_wx) / sum_w

    # Calculate uncertainties in slope (Delta m) and intercept (Delta b)
    delta_m = np.sqrt(1 / (sum_wxx - (sum_wx ** 2) / sum_w))
    delta_b = np.sqrt(sum_wxx / (sum_w * (sum_wxx - (sum_wx ** 2) / sum_w)))

    # Calculate chi-squared
    chi_squared = np.sum(((y - (m * x + b)) / error) ** 2)
    # Create plot
    #axes.errorbar(x, y, yerr=error, fmt='o', label='Data points with uncertainties', markersize=5)
    if make_plot:
        axes.plot(x, m * x + b) # label=f'Best-fit line: y = {m:.2f}x + {b:.2f}'
        #axes.fill_between(x, (m - delta_m) * x + (b - delta_b), (m + delta_m) * x + (b + delta_b), color='gray', alpha=0.5)
    return m, b, delta_m, delta_b, chi_squared

def poisson_interval(k, alpha=0.05): 
    """
    uses chisquared info to get the poisson interval. Uses scipy.stats 
    (imports in function). 
    """
    from scipy.stats import chi2
    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
    return low, high

def matching_purity(clusters, q_bins=6, q_range=None, plot_vlines=True, plot_log_scale=False, plot_legend=True, figTitle=None, saveFig=False, fileName=None): 
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    q_io1 = clusters[clusters['io_group'] == 1]['q']*221*1e-3
    q_io2 = clusters[clusters['io_group'] == 2]['q']*221*1e-3
    z_drift_mid_io1 = clusters[clusters['io_group'] == 1]['z_drift_mid']
    z_drift_mid_io2 = clusters[clusters['io_group'] == 2]['z_drift_mid']
    
    plot_bins = q_bins
    if q_range is None:
        q_min_max = [0, 50]
    else:
        q_min_max = [q_range[0], q_range[1]]
    plot_binsize = (q_min_max[1] - q_min_max[0])/plot_bins
    interval = 0.683

    matching_purity_io1, matching_purity_io2 = [], []
    matching_purity_error_io1, matching_purity_error_io2 = [], []
    real_matches_io1, real_matches_io2 = [], []
    real_matches_error_io1, real_matches_error_io2 = [], []

    q_for_plot = []
    q_values = []
    x_mid_values, y_mid_values, z_drift_values = [], [], []

    Range_io2 = [-400, 600]
    Range_io1 = [-600, 400]
    nbins = 150
    
    max_bins_io1 = []
    max_bins_io2 = []
    # loop through charge bins and plot z_reco distribution + calculate matching purity
    for i in tqdm(range(plot_bins)):

        # start and end points to consider for this charge bin
        start = q_min_max[0] + i*plot_binsize
        end = q_min_max[0] + (i+1)*plot_binsize

        # get z points for this selection and particular io group
        mask_io1 = (q_io1 > start) & (q_io1 < end) #& (x_mid_io1 > -280) & (x_mid_io1 < -270)
        z_drift_sel_io1 = z_drift_mid_io1[mask_io1]
        mask_io2 = (q_io2 > start) & (q_io2 < end) #& (x_mid_io2 > -280) & (x_mid_io2 < -270)
        z_drift_sel_io2 = z_drift_mid_io2[mask_io2]

        ### make TPC 2 histogram
        #if i < 1000:
        #ax0 = axes[1].hist(z_drift_sel_io2, bins=nbins, range=(Range_io2[0], Range_io2[1]), label=f'{(start / (1/221) * 1e-3):.2f} < q < {(end / (1/221) * 1e-3):.2f} ke-', histtype='step')
        ax0 = axes[0][1].hist(z_drift_sel_io2, bins=nbins, range=(Range_io2[0], Range_io2[1]), label=f'{start:.2f} < q < {end:.2f} ke-', histtype='step')
        bincenters_io2, bincontents_io2 = ax0[1], ax0[0]
        max_bins_io2.append(np.max(bincontents_io2))
        TPC2_errorbars = np.zeros((2, len(bincontents_io2)))
        for j, C in enumerate(bincontents_io2):
            TPC2_errorbars[:, j] = np.abs(np.array(list(poisson_interval(C, alpha=1-interval))) - C)
        axes[0][1].errorbar(ax0[1][:-1] + 0.5*(Range_io2[1] - Range_io2[0])/nbins, ax0[0], yerr=TPC2_errorbars,color='k',fmt='o',markersize = 0.5, linewidth=0.5)
        axes[0][1].set_xlim(-240, 550)
        
        ### make TPC1 histogram
        #ax1 = axes[0].hist(z_drift_sel_io1, bins=nbins, range=(Range_io1[0], Range_io1[1]), label=f'{(start / (1/221) * 1e-3):.2f} < q < {(end / (1/221) * 1e-3):.2f} ke-', histtype='step')
        ax1 = axes[0][0].hist(z_drift_sel_io1, bins=nbins, range=(Range_io1[0], Range_io1[1]), label=f'{start:.2f} < q < {end:.2f} ke-', histtype='step')
        bincenters_io1, bincontents_io1 = ax1[1], ax1[0]
        max_bins_io1.append(np.max(bincontents_io1))
        TPC1_errorbars = np.zeros((2, len(bincontents_io1)))
        for j,C in enumerate(bincontents_io1):
            TPC1_errorbars[:, j] = np.abs(np.array(list(poisson_interval(C, alpha=1-interval))) - C)
        axes[0][0].errorbar(ax1[1][:-1] + 0.5*(Range_io1[1] - Range_io1[0])/nbins, ax1[0], yerr=TPC1_errorbars,color='k',fmt='o',markersize = 0.5, linewidth=0.5)
        axes[0][0].set_xlim(-550, 240)
        
        for k in range(0,2):
            if k == 0:
                LSB_min, LSB_max = -525, -325
                SR_min, SR_max = -310.31, 0
                bincenters = bincenters_io1 
                bincontents = bincontents_io1
            elif k == 1:
                LSB_min, LSB_max = 325, 525
                SR_min, SR_max = 0, 310.31
                bincenters = bincenters_io2
                bincontents = bincontents_io2
            # calculate errorbars in lower side band
            LSB_mask = ((bincenters[:-1] >= LSB_min) & (bincenters[:-1] <= LSB_max))
            LSB_errorbars = np.zeros((2, np.sum(LSB_mask)))
            for j,C in enumerate(bincontents[LSB_mask]):
                LSB_errorbars[:, j] = np.abs(np.array(list(poisson_interval(C, alpha=1-interval))) - C)
            LSB_sum = np.sum(bincontents[LSB_mask])

            ### calculate errorbars in signal region for TPC2
            SR_mask = (bincenters[:-1] >= SR_min) & (bincenters[:-1] <= SR_max)
            SR_sum = np.sum(bincontents[SR_mask])
            SR_errorbars = np.zeros((2, np.sum(SR_mask)))
            for j,C in enumerate(bincontents[SR_mask]):
                SR_errorbars[:, j] = np.abs(np.array(list(poisson_interval(C, alpha=1-interval))) - C)
            SR_sum_errorbars = np.sqrt( np.sum( SR_errorbars**2 , axis=1) )

            a = np.max(bincenters[:-1][SR_mask]) - np.min(bincenters[:-1][SR_mask]) # time range in SR
            b = np.max(bincenters[:-1][LSB_mask]) - np.min(bincenters[:-1][LSB_mask]) # time range in LSB
            S = SR_sum - (a/b)*LSB_sum # est signal counts in SR
            sigma_N = np.sqrt(np.sum( SR_errorbars**2 , axis=1)) # total error on SR
            sigma_M = np.sqrt(np.sum( LSB_errorbars**2 , axis=1)) # total error on LSB
            sigma_B = (a/b)*sigma_M # total error on LSB scaled to SR
            sigma_S =  np.sqrt( sigma_N**2 + sigma_B**2 ) # error on est real matches in SR

            P = 1 - (a/b)*LSB_sum/SR_sum
            P_errorbars = (a/b) * LSB_sum/SR_sum * np.sqrt( ( sigma_M / LSB_sum )**2 + ( SR_sum_errorbars / SR_sum )**2 )

            if k == 0:
                matching_purity_io1.append(P)
                matching_purity_error_io1.append(P_errorbars)
                real_matches_io1.append(S)
                real_matches_error_io1.append(sigma_S)
            elif k == 1:
                matching_purity_io2.append(P)
                matching_purity_error_io2.append(P_errorbars)
                real_matches_io2.append(S)
                real_matches_error_io2.append(sigma_S)
                
        q_for_plot.append((end+start)/2)
    if plot_vlines:
        axes[0][1].vlines(0, ymin=0, ymax=max(max_bins_io2)*1.2, color='y', label='cathode',linewidth=1)
        axes[0][1].vlines(304.31, ymin=0, ymax=max(max_bins_io2)*1.2, color='r', label='anode',linewidth=1)
        axes[0][0].vlines(0, ymin=0, ymax=max(max_bins_io1)*1.2, color='y', label='cathode',linewidth=1)
        axes[0][0].vlines(-304.31, ymin=0, ymax=max(max_bins_io1)*1.2, color='r', label='anode',linewidth=1)
    
    if plot_log_scale:
        axes[0][1].set_yscale('log')
        axes[0][0].set_yscale('log')
        
    if plot_legend:
        axes[0][1].legend(fontsize=4.5, loc='upper left')
        axes[0][0].legend(fontsize=4.5, loc='upper left')
    
    axes[0][0].set_xlabel(r'Reconstructed Drift Coordinate [mm]')
    axes[0][1].set_xlabel(r'Reconstructed Drift Coordinate [mm]') 
    axes[0][1].set_ylabel('Counts')
    axes[0][0].set_ylabel('Counts')
    
    axes[0][0].set_title('TPC1')
    axes[0][1].set_title('TPC2')
    #axes[0][0].text(0.85, 0.89, 'TPC1', transform=axes[0][0].transAxes,
    #     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    #axes[0][1].text(0.85, 0.89, 'TPC2', transform=axes[0][1].transAxes,
    #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    axes[1][0].set_ylabel('Purity Fraction')
    axes[1][1].set_ylabel('Purity Fraction')
    axes[1][0].plot(q_for_plot, matching_purity_io1, 'bo', markersize=3,label='TPC1')
    if plot_bins > 1:
        xerr=np.ones_like(q_for_plot)*(q_for_plot[1] - q_for_plot[0])/2
    else:
        xerr=None
    axes[1][0].errorbar(q_for_plot, matching_purity_io1, xerr=xerr,yerr=np.array(matching_purity_error_io1).transpose(),color='k',fmt='o',markersize = 0.5, linewidth=1)
    #fig.suptitle('Purity Fraction of Real Charge-Light Matched Charge Clusters \n (module-1, 5 hrs of data, 2022_02_08)')
    axes[1][1].set_xlabel('Charge [ke-]')
    axes[1][0].set_xlabel('Charge [ke-]')
    axes[1][1].plot(q_for_plot, matching_purity_io2, 'bo', markersize=3, label='TPC2')
    axes[1][1].errorbar(q_for_plot, matching_purity_io2, xerr=xerr,yerr=np.array(matching_purity_error_io2).transpose(),color='k',fmt='o',markersize = 0.5, linewidth=1)
    fig.suptitle(figTitle)
    if saveFig:
        plt.savefig(fileName)
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
    

