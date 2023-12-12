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
