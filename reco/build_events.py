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

def build_charge_events_small_clusters(labels,dataword,txyz,v_ref,v_cm,v_ped,gain,\
                                       unix,io_group,event_dtype,hits_size,hits_dtype):
    ### Build charge events by adding up packet charge from individual DBSCAN clusters
    # Inputs: 
    #   labels_noise_list: list of noise labels from DBSCAN
    #   dataword: packet ADC counts
    #   unique_ids: unique id for each pixel corresponding to the packets
    #   v_ref, v_cm, v_ped, gain: arrays providing pixel parameters for ADC->ke- conversion
    # Outputs:
    #   results: array containing event information

    charge = adcs_to_ke(dataword, v_ref,v_cm,v_ped,gain)
    q_vals = np.bincount(labels, weights=charge)
    io_group_vals = np.bincount(labels, weights=io_group)
    txyz = np.array(txyz)
    
    hits = np.zeros((0,), dtype=hits_dtype)
    
    t_vals_all = txyz[:,0]
    x_vals_all = txyz[:,1]
    y_vals_all = txyz[:,2]
    z_vals_all = txyz[:,3]
    io_group_vals_all = io_group
    q_vals_all = charge
    unix_vals_all = unix
    # loop through unique labels (charge events) to make hits dataset
    unique_labels = np.unique(labels)
    for i in range(hits_size, len(unique_labels)+hits_size):
        label = unique_labels[i - hits_size]
        label_mask = labels == label
        t_vals_event = t_vals_all[label_mask]
        x_vals_event = x_vals_all[label_mask]
        y_vals_event = y_vals_all[label_mask]
        z_vals_event = z_vals_all[label_mask]
        io_group_vals_event = io_group_vals_all[label_mask]
        q_vals_event = q_vals_all[label_mask]
        unix_vals_event = unix_vals_all[label_mask]
        
        # loop through hits within each event, concatenate to hits array with cluster index
        nhits = np.sum(label_mask)
        hits_event = np.zeros((nhits,), dtype=hits_dtype)
        hits_event['q'] = q_vals_event
        hits_event['io_group'] = io_group_vals_event
        hits_event['t'] = t_vals_event/(v_drift*1e1) * 1e3
        hits_event['x'] = x_vals_event
        hits_event['y'] = y_vals_event
        hits_event['z'] = z_vals_event
        hits_event['unix'] = unix_vals_event
        hits_event['cluster_index'] = np.ones(nhits)*i
        hits = np.concatenate((hits, hits_event))
    
    # find midpoint of clustered hits and save array of all event information
    n_vals = np.bincount(labels)
    # get rid of n_vals that are 0, which are a natural result of bincount, 
    # otherwise we get divide by 0 later
    t_vals = np.bincount(labels, weights=txyz[:,0].astype('i8'))[n_vals != 0] # add up x values of hits in cluster then avg
    unix_vals = np.bincount(labels, weights=unix)[n_vals != 0]
    q_vals = q_vals[n_vals != 0]
    n_vals = n_vals[n_vals != 0]
    
    results = np.zeros((len(n_vals[n_vals != 0]),), dtype=event_dtype)
    results['nhit'] = n_vals
    results['q'] = q_vals
    results['t'] = (t_vals/(n_vals*v_drift*1e1) * 1e3).astype('i8') # average for each event
    results['unix'] = (unix_vals/n_vals).astype('i8') # all of these hits should have the same unix anyway
    results['io_group'] = (io_group_vals/n_vals).astype('i4')
    results['matched'] = np.zeros(len(n_vals), dtype='i4')
    results['light_index'] = np.ones(len(n_vals), dtype='i4')*-1
    return results, hits

def build_charge_events_large_clusters(labels,dataword,txyz,v_ref,v_cm,v_ped,gain,unix,io_group,event_dtype,\
                                      hits_size, hits_dtype):
    ### Build charge events by adding up packet charge from individual DBSCAN clusters
    # Inputs: 
    #   labels_noise_list: list of noise labels from DBSCAN
    #   dataword: packet ADC counts
    #   unique_ids: unique id for each pixel corresponding to the packets
    #   v_ref, v_cm, v_ped, gain: arrays providing pixel parameters for ADC->ke- conversion
    # Outputs:
    #   results: array containing event information

    charge = adcs_to_ke(dataword, v_ref,v_cm,v_ped,gain)
    q_vals = np.bincount(labels, weights=charge)
    txyz = np.array(txyz)
    
    hits = np.zeros((0,), dtype=hits_dtype)
    
    t_vals_all = txyz[:,0]
    x_vals_all = txyz[:,1]
    y_vals_all = txyz[:,2]
    z_vals_all = txyz[:,3]
    io_group_vals_all = io_group
    q_vals_all = charge
    unix_vals_all = unix
    # loop through unique labels (charge events) to make hits dataset
    unique_labels = np.unique(labels)
    for i in range(hits_size, len(unique_labels)+hits_size):
        label = unique_labels[i - hits_size]
        label_mask = labels == label
        t_vals_event = t_vals_all[label_mask]
        x_vals_event = x_vals_all[label_mask]
        y_vals_event = y_vals_all[label_mask]
        z_vals_event = z_vals_all[label_mask]
        io_group_vals_event = io_group_vals_all[label_mask]
        q_vals_event = q_vals_all[label_mask]
        unix_vals_event = unix_vals_all[label_mask]
        
        # loop through hits within each event, concatenate to hits array with cluster index
        nhits = np.sum(label_mask)
        hits_event = np.zeros((nhits,), dtype=hits_dtype)
        hits_event['q'] = q_vals_event
        hits_event['io_group'] = io_group_vals_event
        hits_event['t'] = t_vals_event/(v_drift*1e1) * 1e3
        hits_event['x'] = x_vals_event
        hits_event['y'] = y_vals_event
        hits_event['z'] = z_vals_event
        hits_event['unix'] = unix_vals_event
        hits_event['cluster_index'] = np.ones(nhits)*i
        hits = np.concatenate((hits, hits_event))
    
    timestamps = txyz[:,0]
    indices_sorted = np.argsort(labels)
    labels = labels[indices_sorted]
    timestamps = timestamps[indices_sorted]
    label_indices = np.concatenate(([0], np.flatnonzero(labels[:-1] != labels[1:])+1, [len(labels)]))
    label_timestamps = np.split(timestamps, label_indices[1:-1])
    
    min_timestamps = np.array([np.min(t) for t in label_timestamps], dtype='i8')
    max_timestamps = np.array([np.max(t) for t in label_timestamps], dtype='i8')

    # save array of event information
    n_vals = np.bincount(labels)
    #t_vals = np.bincount(labels, weights=txyz[:,0])[n_vals != 0] # add up x values of hits in cluster then avg
    io_group_vals = np.bincount(labels, weights=io_group)
    unix_vals = np.bincount(labels, weights=unix)[n_vals != 0]
    q_vals = q_vals[n_vals != 0]
    n_vals = n_vals[n_vals != 0] # get rid of n_vals that are 0, otherwise get divide by 0 later

    #q_vals_not_0 = q_vals != 0
    #t_vals = t_vals[q_vals_not_0]
    #n_vals = n_vals[q_vals_not_0]
    #q_vals = q_vals[q_vals_not_0]
    #unix_vals = unix_vals[q_vals_not_0]
    #io_group_vals = io_group_vals[q_vals_not_0]
    
    results = np.zeros((len(n_vals[n_vals != 0]),), dtype=event_dtype)
    results['nhit'] = n_vals
    results['q'] = q_vals
    results['unix'] = (unix_vals/n_vals).astype('i8') # all of these hits should have the same unix anyway
    results['io_group'] = (io_group_vals/n_vals).astype('i4')
    results['t_min'] = min_timestamps/(v_drift*1e1) * 1e3
    results['t_max'] = max_timestamps/(v_drift*1e1) * 1e3
    results['matched'] = np.zeros(len(n_vals), dtype='i4')
    results['light_index'] = np.ones(len(n_vals), dtype='i4')*-1
    return results, hits

def analysis(packets, pixel_xy, mc_assn, detector,hits_small_clusters_max_cindex,hits_large_clusters_max_cindex):
    #if mc_assn != None:
    #    mc_assn = mc_assn[packets['packet_type'] != 6]
    #packets = packets[packets['packet_type'] != 6]
    
    packet_type = packets['packet_type']
    pkt_7_mask = packet_type == 7
    pkt_4_mask = packet_type == 4
    pkt_0_mask = packet_type == 0
    # grab the PPS timestamps of pkt type 7s and correct for PACMAN clock drift
    PPS_pt7 = PACMAN_drift(packets, detector)[pkt_7_mask].astype('i8')*1e-1*1e3 # ns
    # assign a unix timestamp to each packet based on the timestamp of the previous packet type 4
    timestamps = packets['timestamp'].astype('i8')
    unix_timestamps = timestamps
    unix_timestamps[np.invert(pkt_4_mask)] = 0
    nonzero_indices = np.nonzero(unix_timestamps)[0]
    unix_timestamps = np.interp(np.arange(len(unix_timestamps)), nonzero_indices, unix_timestamps[nonzero_indices])
    unix_pt7 = unix_timestamps[pkt_7_mask].astype('i8')
    unix = unix_timestamps[pkt_0_mask].astype('i8')
    
    # apply a few PPS timestamp corrections, and select only data packets for analysis
    ts, packets, mc_assn, unix = timestamp_corrector(packets, mc_assn, unix, detector)
    dataword = packets['dataword']
    io_group = packets['io_group']
    
    # zip up y, z, and t values for clustering
    txyz = zip_pixel_tyz(packets,ts, pixel_xy)
    v_ped, v_cm, v_ref, gain = calibrations(packets, mc_assn, detector)
    
    # cluster packets to find track-like charge events
    txyz_core_tracks, txyz_noise_tracks, txyz_noncore_tracks, db = cluster_packets(eps_tracks, min_samples_tracks, txyz)
    labels_tracks = db.labels_
    
    # cluster packets that remain after tracks have been removed
    txyz_core_small_clusters, txyz_noise_small_clusters, txyz_noncore_small_clusters, db_small_clusters = \
        cluster_packets(eps_noise, min_samples_noise,txyz_noise_tracks)
    
    noise_samples_mask = db.labels_ == -1
    labels_small_clusters = db_small_clusters.labels_
    dataword_small_clusters = np.array(dataword)[noise_samples_mask]
    v_ref_small_clusters = v_ref[noise_samples_mask]
    v_cm_small_clusters = v_cm[noise_samples_mask]
    v_ped_small_clusters = v_ped[noise_samples_mask]
    gain_small_clusters = gain[noise_samples_mask]
    unix_small_clusters = unix[noise_samples_mask]
    io_group_small_clusters = io_group[noise_samples_mask]
    
    # build charge events
    event_small_clusters_dtype = np.dtype([('nhit', '<i4'), ('q', '<f8'),('io_group', 'i4'),\
                                           ('t', '<i8'),('unix', 'i8'),\
                                           ('matched', 'i4'), ('light_index', 'i4')])
    hits_dtype = np.dtype([('q', '<f8'),('io_group', '<i4'),\
                            ('t', '<i8'),('x', '<f8'), ('y', '<f8'), ('z', '<f8'),\
                            ('unix', '<i8'), ('cluster_index', '<i4')])
    results_small_clusters, hits_small_clusters = \
        build_charge_events_small_clusters(labels_small_clusters,dataword_small_clusters,txyz_noise_tracks,\
         v_ref=v_ref_small_clusters,v_cm=v_cm_small_clusters,v_ped=v_ped_small_clusters,\
         gain=gain_small_clusters, unix=unix_small_clusters, io_group=io_group_small_clusters,\
         event_dtype=event_small_clusters_dtype,hits_size=hits_small_clusters_max_cindex,\
         hits_dtype=hits_dtype)

    noise_samples_mask_inverted = np.invert(noise_samples_mask)
    labels_large_clusters = labels_tracks[noise_samples_mask_inverted]
    dataword_large_clusters = np.array(dataword)[noise_samples_mask_inverted]
    txyz_large_clusters = np.array(txyz)[noise_samples_mask_inverted]
    v_ref_large_clusters = v_ref[noise_samples_mask_inverted]
    v_cm_large_clusters = v_cm[noise_samples_mask_inverted]
    v_ped_large_clusters = v_ped[noise_samples_mask_inverted]
    gain_large_clusters = gain[noise_samples_mask_inverted]
    unix_large_clusters = unix[noise_samples_mask_inverted]
    io_group_large_clusters = io_group[noise_samples_mask_inverted]
    
    event_large_clusters_dtype = np.dtype([('nhit', '<i4'), ('q', '<f8'),('io_group', '<i4'),\
                            ('t_max', '<i8'), ('t_min', '<i8'),\
                            ('unix', '<i8'), ('matched', '<i4'), ('light_index', '<i4')])
    
    if np.size(labels_large_clusters) > 0:
        results_large_clusters, hits_large_clusters  = \
            build_charge_events_large_clusters(labels_large_clusters,dataword_large_clusters,txyz_large_clusters,\
            v_ref=v_ref_large_clusters,v_cm=v_cm_large_clusters,v_ped=v_ped_large_clusters,\
            gain=gain_large_clusters, unix=unix_large_clusters, io_group=io_group_large_clusters,\
            event_dtype=event_large_clusters_dtype,hits_size=hits_large_clusters_max_cindex,\
            hits_dtype=hits_dtype)
    else:
        results_large_clusters = np.zeros((0,), dtype=event_large_clusters_dtype)
        hits_large_clusters = np.zeros((0,), dtype=hits_dtype)

    return results_small_clusters, results_large_clusters, unix_pt7, PPS_pt7, hits_small_clusters, hits_large_clusters
