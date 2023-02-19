import numpy as np
from sklearn.cluster import DBSCAN
import json
import time
from consts import *
from calibrate import *
from preclustering import *
import matplotlib.pyplot as plt
import scipy.stats
import h5py

def cluster_packets(eps,min_samples,txyz):
    ### Cluster packets into charge events
    # INPUT: DBSCAN parameters (eps: mm; min_samples: int), packet txyz list
    # OUTPUT: txyz values for core, noise, and noncore samples. And returns DBSCAN fit db.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(txyz) 
    # core samples
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    txyz_coresamples = np.array(txyz)[core_samples_mask]
    # noise samples
    noise_samples_mask = db.labels_ == -1
    txyz_noise = np.array(txyz)[noise_samples_mask]
    # non-core samples
    coreplusnoise_samples_mask = core_samples_mask + noise_samples_mask
    noncore_samples_mask = np.invert(coreplusnoise_samples_mask)
    txyz_noncoresamples = np.array(txyz)[noncore_samples_mask]
    return txyz_coresamples, txyz_noise, txyz_noncoresamples,db

def build_charge_events(labels,dataword,txyz,v_ref,v_cm,v_ped,gain):
    ### Build charge events by adding up packet charge from individual DBSCAN clusters
    # Inputs: 
    #   labels_noise_list: list of noise labels from DBSCAN
    #   dataword: packet ADC counts
    #   unique_ids: unique id for each pixel corresponding to the packets
    #   v_ref, v_cm, v_ped, gain: arrays providing pixel parameters for ADC->ke- conversion
    # Outputs:
    #   nqtxyz: array containing event information

    charge = adcs_to_ke(dataword, v_ref,v_cm,v_ped,gain)
    q_vals = np.bincount(labels, weights=charge)
    txyz = np.array(txyz)
    #print('length of txyz = ', len(txyz))

    # find midpoint of clustered hits and save array of all event information
    n_vals = np.bincount(labels)
    t_vals = np.bincount(labels, weights=txyz[:,0])[n_vals != 0] # add up x values of hits in cluster then avg
    x_vals = np.bincount(labels, weights=txyz[:,1])[n_vals != 0]
    y_vals = np.bincount(labels, weights=txyz[:,2])[n_vals != 0]
    z_vals = np.bincount(labels, weights=txyz[:,3])[n_vals != 0]
    q_vals = q_vals[n_vals != 0]
    n_vals = n_vals[n_vals != 0] # get rid of n_vals that are 0, otherwise get divide by 0 later
    
    q_vals_not_0 = q_vals != 0
    t_vals = t_vals[q_vals_not_0]
    x_vals = x_vals[q_vals_not_0]
    y_vals = y_vals[q_vals_not_0]
    z_vals = z_vals[q_vals_not_0]
    n_vals = n_vals[q_vals_not_0]
    q_vals = q_vals[q_vals_not_0]
    
    # for track-like events
    #labels_tracks = np.array(labels_cosmics)
    #labels_tracks = labels_cosmics[labels_cosmics != -1]
    #n_vals_tracks = np.bincount(labels_cosmics)
    
    event_dtype = np.dtype([('nhit', '<u1'), ('q', '<f8'),('t', '<f8'),('x', '<f8'),('y', '<f8'),('z', '<f8')])
    results = np.zeros((len(n_vals[n_vals != 0]),), dtype=event_dtype)
    results['nhit'] = n_vals
    results['q'] = q_vals
    results['t'] = t_vals/n_vals # average for each event
    results['x'] = x_vals/n_vals
    results['y'] = y_vals/n_vals
    results['z'] = z_vals/n_vals
    return results

def analysis(packets, pixel_xy, mc_assn):
    ts, packets = timestamp_corrector(packets, mc_assn)
    dataword = packets['dataword']
    # zip up y, z, and t values for clustering
    txyz = zip_pixel_tyz(packets,ts, pixel_xy)
    v_ped, v_cm, v_ref, gain = calibrations(packets, mc_assn)

    # cluster packets to find track-like charge events
    #### this block is slow
    #if mc_assn == None:
    txyz_core_tracks, txyz_noise_tracks, txyz_noncore_tracks, db = cluster_packets(eps_tracks, min_samples_tracks, txyz)
    labels_tracks = db.labels_
    
    # cluster packets that remain after tracks have been removed
    txyz_core_small_clusters, txyz_noise_small_clusters, txyz_noncore_small_clusters, db_small_clusters = \
        cluster_packets(eps_noise, min_samples_noise,txyz_noise_tracks)
    noise_samples_mask = db.labels_ == -1
    #tracks_samples_mask = db.labels != -1
    #else: # FIXME: Should do the track-level clustering here too. 
    #    txyz_core_MC, txyz_noise_MC, txyz_noncore_MC, db_MC = cluster_packets(eps_noise, min_samples_noise,txyz)
    #    labels_small_clusters = db_MC.labels_
    #    txyz_noise = txyz
    #    noise_samples_mask = db_MC.labels_ == -1
    
    #if mc_assn == None:
    #    dataword = np.array(dataword)[noise_samples_mask]
    #    v_ref = v_ref[noise_samples_mask]
    #    v_cm = v_cm[noise_samples_mask]
    #    v_ped = v_ped[noise_samples_mask]
    #    gain = gain[noise_samples_mask]
    #else:
    #    dataword = np.array(dataword)
    labels_small_clusters = db_small_clusters.labels_
    dataword_small_clusters = np.array(dataword)[noise_samples_mask]
    v_ref_small_clusters = v_ref[noise_samples_mask]
    v_cm_small_clusters = v_cm[noise_samples_mask]
    v_ped_small_clusters = v_ped[noise_samples_mask]
    gain_small_clusters = gain[noise_samples_mask]
    
    # build charge events
    results_small_clusters = build_charge_events(labels_small_clusters,dataword_small_clusters,txyz_noise_tracks,\
                v_ref=v_ref_small_clusters,v_cm=v_cm_small_clusters,v_ped=v_ped_small_clusters,gain=gain_small_clusters)
    
    noise_samples_mask_inverted = np.invert(noise_samples_mask)
    labels_large_clusters = labels_tracks[noise_samples_mask_inverted]
    dataword_large_clusters = np.array(dataword)[noise_samples_mask_inverted]
    txyz_large_clusters = np.array(txyz)[noise_samples_mask_inverted]
    v_ref_large_clusters = v_ref[noise_samples_mask_inverted]
    v_cm_large_clusters = v_cm[noise_samples_mask_inverted]
    v_ped_large_clusters = v_ped[noise_samples_mask_inverted]
    gain_large_clusters = gain[noise_samples_mask_inverted]
    
    results_large_clusters = build_charge_events(labels_large_clusters,dataword_large_clusters,txyz_large_clusters,\
                v_ref=v_ref_large_clusters,v_cm=v_cm_large_clusters,v_ped=v_ped_large_clusters,gain=gain_large_clusters)

    return results_small_clusters, results_large_clusters
