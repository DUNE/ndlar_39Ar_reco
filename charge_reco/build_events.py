import numpy as np
from sklearn.cluster import DBSCAN
import json
import time
import consts
from calibrate import *
from preclustering import *
import matplotlib.pyplot as plt
import h5py

def cluster_packets(eps,min_samples,txyz):
    ### Cluster packets into charge events
    # INPUT: DBSCAN parameters (eps: mm; min_samples: int), packet txyz list
    # OUTPUT: DBSCAN fit db.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(txyz) 
    return db

def getEventIDs(txyz, mc_assn, tracks, event_ids):
    for i in range(len(txyz)):
        index = int(mc_assn[i][0][0])
        tracks_index = tracks[index]
        event_id = tracks_index[consts.EVENT_SEPARATOR]
        event_ids[i] = event_id

def find_charge_clusters(labels,dataword,txyz,v_ref,v_cm,v_ped,unix,io_group,unique_ids, \
                                      hits_size, mc_assn, tracks):
    ### Make hits and clusters datasets from DBSCAN clusters and corresponding hits
    # Inputs: 
    #   labels: list of labels from DBSCAN
    #   dataword: packet ADC counts
    #   unique_ids: unique id for each pixel corresponding to the packets
    #   v_ref, v_cm, v_ped: arrays providing pixel parameters for ADC->ke- conversion
    #   ...
    # Outputs:
    #   clusters: array of cluster data
    #   hits: array of hit-level data
    labels = labels
    indices_sorted = np.argsort(labels)
    labels = labels[indices_sorted]
    txyz = txyz[indices_sorted]
    v_ref = v_ref[indices_sorted]
    v_cm = v_cm[indices_sorted]
    v_ped = v_ped[indices_sorted]
    unix = unix[indices_sorted]
    io_group = io_group[indices_sorted]
    unique_ids = unique_ids[indices_sorted]
    dataword = dataword[indices_sorted]
    
    charge = adcs_to_mV(dataword, v_ref, v_cm, v_ped)
    q_vals = np.bincount(labels, weights=charge)
    
    # get event IDs if MC
    event_ids = np.zeros(len(txyz))
    if mc_assn is not None:
        getEventIDs(txyz, mc_assn, tracks, event_ids)
    else:
        event_ids = np.ones_like(len(txyz))*-1
    
    # add hits to hits dataset
    unique_labels = np.unique(labels)
    hits = np.zeros((np.size(labels),), dtype=consts.hits_dtype)
    hits['q'] = charge
    hits['io_group'] = io_group
    hits['t'] = (txyz[:,0]/(v_drift*1e1) * 1e3).astype('i8')
    hits['x'] = txyz[:,1]
    hits['y'] = txyz[:,2]
    hits['z_anode'] = txyz[:,3]
    hits['z_drift'] = txyz[:,3]
    hits['unique_id'] = unique_ids
    hits['unix'] = unix
    labels_global = np.copy(labels)
    labels_global += hits_size
    hits['cluster_index'] = labels_global
    hits['event_id'] = event_ids
    hits['light_trig_id'] = np.ones(len(labels), dtype='i4')*-1
        
    label_indices = np.concatenate(([0], np.flatnonzero(labels[:-1] != labels[1:])+1, [len(labels)]))
    label_timestamps = np.split(txyz[:,0]/(v_drift*1e1) * 1e3, label_indices[1:-1])
    label_x = np.split(txyz[:,1], label_indices[1:-1])
    label_y = np.split(txyz[:,2], label_indices[1:-1])
    label_z = np.split(txyz[:,3], label_indices[1:-1])
    
    min_timestamps = np.array(list(map(np.min, label_timestamps)), dtype='i8')
    max_timestamps = np.array(list(map(np.max, label_timestamps)), dtype='i8')
    
    # save array of event information
    n_vals = np.bincount(labels)
    io_group_vals = np.bincount(labels, weights=io_group)[n_vals != 0]
    unix_vals = np.bincount(labels, weights=unix)[n_vals != 0]
    q_vals = q_vals[n_vals != 0]
    n_vals = n_vals[n_vals != 0] # get rid of n_vals that are 0, otherwise get divide by 0 later
    
    x_min = list(map(np.min, label_x))
    x_max = list(map(np.max, label_x))
    y_min = list(map(np.min, label_y))
    y_max = list(map(np.max, label_y))
    z_min = list(map(np.min, label_z))
    z_max = list(map(np.max, label_z))
    
    clusters = np.zeros((len(n_vals),), dtype=consts.clusters_dtype)
    clusters['nhit'] = n_vals
    clusters['q'] = q_vals
    clusters['unix'] = (unix_vals/n_vals).astype('i8') # all of these hits should have the same unix anyway
    clusters['io_group'] = (io_group_vals/n_vals).astype('i4')
    clusters['t_min'] = min_timestamps
    clusters['t_mid'] = ((min_timestamps + max_timestamps)/2).astype('i8')
    clusters['t_max'] = max_timestamps
    clusters['x_min'] = np.array(x_min, dtype='f8')
    clusters['x_max'] = np.array(x_max, dtype='f8')
    clusters['x_mid'] = ((np.array(x_min) + np.array(x_max))/2).astype('f8')
    clusters['y_min'] = np.array(y_min, dtype='f8')
    clusters['y_mid'] = ((np.array(y_min) + np.array(y_max))/2).astype('f8')
    clusters['y_max'] = np.array(y_max, dtype='f8')
    clusters['z_min'] = np.array(z_min, dtype='f8')
    clusters['z_mid'] = ((np.array(z_min) + np.array(z_max))/2).astype('f8')
    clusters['z_max'] = np.array(z_max, dtype='f8')
    clusters['z_drift_min'] = np.ones(len(n_vals), dtype='f8')*-1
    clusters['z_drift_mid'] = np.ones(len(n_vals), dtype='f8')*-1
    clusters['z_drift_max'] = np.ones(len(n_vals), dtype='f8')*-1
    clusters['matched'] = np.zeros(len(n_vals), dtype='i4')*-1
    clusters['ext_trig_index'] = np.ones(len(n_vals), dtype='i4')*-1
    clusters['light_index'] = np.ones(len(n_vals), dtype='i4')*-1
    clusters['t0'] = np.ones(len(n_vals), dtype='i4')*-1
    clusters['light_trig_id'] = np.ones(len(n_vals), dtype='i4')*-1
    return clusters, hits

def analysis(packets,pixel_xy,mc_assn,tracks,module,hits_max_cindex):
    ## do charge reconstruction
    clusters = np.zeros((0,), dtype=consts.clusters_dtype)
    hits = np.zeros((0,), dtype=consts.hits_dtype)
        
    pkt_7_mask = packets['packet_type'] == 7
    pkt_4_mask = packets['packet_type'] == 4
    pkt_0_mask = packets['packet_type'] == 0
    
    # grab the PPS timestamps of pkt type 7s and correct for PACMAN clock drift
    PPS_pt7 = PACMAN_drift(packets, module)[pkt_7_mask].astype('i8')*1e-1*1e3 # ns
    io_group_pt7 = packets[pkt_7_mask]['io_group']
    # assign a unix timestamp to each packet based on the timestamp of the previous packet type 4
    unix_timestamps = np.copy(packets['timestamp']).astype('i8')
    unix_timestamps[np.invert(pkt_4_mask)] = 0
    nonzero_indices = np.nonzero(unix_timestamps)[0]
    unix_timestamps = np.interp(np.arange(len(unix_timestamps)), nonzero_indices, unix_timestamps[nonzero_indices])
    unix_pt7 = np.copy(unix_timestamps)[pkt_7_mask].astype('i8')
    unix = np.copy(unix_timestamps)[pkt_0_mask].astype('i8')
    
    ext_trig = np.zeros((np.size(unix_pt7),), dtype=consts.ext_trig_dtype)
    ext_trig['unix'] = unix_pt7
    ext_trig['ts_PPS'] = PPS_pt7
    ext_trig['io_group'] = io_group_pt7
    ext_trig['light_trig_id'] = np.ones_like(io_group_pt7)*-1
    
    # apply a few PPS timestamp corrections, and select only data packets for analysis
    ts, packets, mc_assn, unix = timestamp_corrector(packets, mc_assn, unix, module)
    
    # zip up x, y, z, and t values for clustering
    txyz, packets_keep_mask = zip_pixel_tyz(packets, ts, pixel_xy, module)
    
    # remove packets with key errors
    packets = packets[packets_keep_mask]
    unix = unix[packets_keep_mask]
    
    if mc_assn is not None:
        mc_assn = mc_assn[packets_keep_mask]
        
    v_ped, v_cm, v_ref, unique_ids = calibrations(packets, mc_assn, module)
    db = cluster_packets(eps, min_samples, txyz[:,0:4])
    labels = np.array(db.labels_)
    
    dataword = np.copy(packets['dataword'])
    io_group = np.copy(packets['io_group'])
    
    if np.size(labels) > 0:
        clusters, hits = \
            find_charge_clusters(labels,dataword,txyz,\
            v_ref=v_ref,v_cm=v_cm,v_ped=v_ped,\
            unix=unix, io_group=io_group,\
            unique_ids=unique_ids,\
            hits_size=hits_max_cindex,\
            mc_assn=mc_assn, tracks=tracks)

    return clusters, ext_trig, hits
