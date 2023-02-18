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
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(txyz) 
    txyz = np.array(txyz)
    # core samples
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    txyz_coresamples = np.array(txyz)[core_samples_mask]
    # noise samples
    noise_samples_mask = db.labels_ == -1
    txyz_noise = txyz[noise_samples_mask]
    # non-core samples
    coreplusnoise_samples_mask = core_samples_mask + noise_samples_mask
    noncore_samples_mask = np.invert(coreplusnoise_samples_mask)
    txyz_noncoresamples = txyz[noncore_samples_mask]
    return txyz_coresamples, txyz_noise, txyz_noncoresamples,db

def build_charge_events(labels_noise_list,dataword,txyz,v_ref,v_cm,v_ped,gain):
    ### Build charge events by adding up packet charge from individual DBSCAN clusters
    # Inputs: 
    #   labels_noise_list: list of noise labels from DBSCAN
    #   dataword: packet ADC counts
    #   unique_ids: unique id for each pixel corresponding to the packets
    #   v_ref, v_cm, v_ped, gain: arrays providing pixel parameters for ADC->ke- conversion
    # Outputs:
    #   nqtxyz: array containing event information

    charge = adcs_to_ke(dataword, v_ref,v_cm,v_ped,gain)
    labels_noise = np.array(labels_noise_list)

    q_vals = np.bincount(labels_noise, weights=charge)
    txyz = np.array(txyz)
    #print('length of txyz = ', len(txyz))

    # find midpoint of clustered hits and save array of all event information
    n_vals = np.bincount(labels_noise)

    t_vals = np.bincount(labels_noise, weights=txyz[:,0])[n_vals != 0] # add up x values of hits in cluster then avg
    x_vals = np.bincount(labels_noise, weights=txyz[:,1])[n_vals != 0]
    y_vals = np.bincount(labels_noise, weights=txyz[:,2])[n_vals != 0]
    z_vals = np.bincount(labels_noise, weights=txyz[:,3])[n_vals != 0]
    q_vals = q_vals[n_vals != 0]
    n_vals = n_vals[n_vals != 0] # get rid of n_vals that are 0, otherwise get divide by 0 later
    
    t_vals = t_vals[q_vals != 0]
    x_vals = x_vals[q_vals != 0]
    y_vals = y_vals[q_vals != 0]
    z_vals = z_vals[q_vals != 0]
    n_vals = n_vals[q_vals != 0]
    q_vals = q_vals[q_vals != 0]
    
    # for track-like events
    #labels_tracks = np.array(labels_cosmics)
    #labels_tracks = labels_cosmics[labels_cosmics != -1]
    #n_vals_tracks = np.bincount(labels_cosmics)
    
    event_dtype = np.dtype([('nhit', '<u1'), ('q', '<f8'),('t', '<f8'),('x', '<f8'),('y', '<f8'),('z', '<f8')])
    nqtxyz = np.zeros((len(n_vals[n_vals != 0]),), dtype=event_dtype)
    nqtxyz['nhit'] = n_vals
    nqtxyz['q'] = q_vals
    nqtxyz['t'] = t_vals/n_vals # average for each event
    nqtxyz['x'] = x_vals/n_vals
    nqtxyz['y'] = y_vals/n_vals
    nqtxyz['z'] = z_vals/n_vals
    return nqtxyz

def analysis(file,pixel_xy,sel_start=0, sel_end=-1):
    ts, packets, mc_assn = getPackets(file, sel_start, sel_end)
    dataword = packets['dataword']
    # zip up y, z, and t values for clustering
    txyz = zip_pixel_tyz(packets,ts, pixel_xy)
    v_ped, v_cm, v_ref, gain = calibrations(packets, mc_assn)

    # cluster packets to find track-like charge events
    labels_cosmics, labels_noise, labels_noise_list, accepted_candidates_mask,txyz_noise = 0,0,0,0,0
    #### this block is slow
    if mc_assn == None:
        txyz_core, txyz_noise, txyz_noncore, db = cluster_packets(eps_tracks, min_samples_tracks, txyz)
        labels_cosmics = db.labels_
        
        # packet coordinates for DBSCAN noise
        candidate_t = txyz_noise[:,0]
        candidate_x = txyz_noise[:,1]
        candidate_y = txyz_noise[:,2]
        candidate_z = txyz_noise[:,3]
        
        # cluster packets that remain after tracks have been removed
        txyz_core, txyz_noise_2, txyz_noncore, db_noise = cluster_packets(eps_noise, min_samples_noise,txyz_noise)
        labels_noise = db_noise.labels_
        labels_noise_list = list(labels_noise)
        noise_samples_mask = db.labels_ == -1
        #tracks_samples_mask = db.labels != -1
    else:
        txyz_core, txyz_noise_2, txyz_noncore, db_noise = cluster_packets(eps_noise, min_samples_noise,txyz)
        labels_noise = db_noise.labels_
        labels_noise_list = list(labels_noise)
        txyz_noise = txyz
        #noise_samples_mask = db.labels_ == -1
    
    if mc_assn == None:
        labels_noise = np.array(labels_noise_list)
        dataword = np.array(dataword)[noise_samples_mask]
        #dataword_tracks = np.array(dataword)[tracks_samples_mask]
        v_ref = v_ref[noise_samples_mask]
        v_cm = v_cm[noise_samples_mask]
        v_ped = v_ped[noise_samples_mask]
        gain = gain[noise_samples_mask]
    else:
        dataword = np.array(dataword)
    
    # build charge events out of 39Ar candidates
    results = build_charge_events(labels_noise,dataword,txyz_noise,\
                v_ref=v_ref,v_cm=v_cm,v_ped=v_ped,gain=gain)

    return results
