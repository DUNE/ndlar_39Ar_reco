import numpy as np
from sklearn.cluster import DBSCAN
from larndsim import consts, fee
from collections import defaultdict
import json
import time
from reco_constants import *
import matplotlib.pyplot as plt
import pickle 
import scipy.stats
import h5py

def load_geom_dict(geom_dict_path):
    with open(geom_dict_path, "rb") as f_geom_dict:
        geom_dict = pickle.load(f_geom_dict)
    return geom_dict

def zip_pixel_tyz(packets,ts, pixel_xy):
    ### pre-clustering step
    v_drift = 1.6 # mm/us
    ts_inmm = list(v_drift*ts*0.1) # timestamp in us * drift speed in mm/us
    # put x, y and z pixel plane positions of data packets into lists
    x_inmm, y_inmm,z_inmm = [],[],[]
    num_keyerrors = 0
    num_notkeyerrors = 0
    
    for i in range(len(packets)):
        try:
            xyz = pixel_xy[packets['io_group'][i],packets['io_channel'][i],packets['chip_id'][i],packets['channel_id'][i]]
            x_inmm.append(xyz[0])
            y_inmm.append(xyz[1])
            z_inmm.append(xyz[2])
            num_notkeyerrors += 1
        except:
            x_inmm.append(0.0)
            y_inmm.append(0.0)
            z_inmm.append(0.0)
            num_keyerrors += 1

    #print('Number of keyerrors = ' , num_keyerrors)
    #print('Number of not keyerrors = ' , num_notkeyerrors)
    #if num_notkeyerrors != 0:
    #    print('Fraction of keyerrors for packets = ', num_keyerrors/num_notkeyerrors)
    #unique_id = (((packets['io_group'][i].astype(int)) * 256
    #                    + packets['io_channel'][i].astype(int)) * 256
    #                    + packets['chip_id'][i].astype(int)) * 64 \
    #                    + packets['channel_id'][i].astype(int)
    
    # zip together t, y, z arrays together to give to clustering method
    txyz = []
    for t,x,y,z in zip(ts_inmm,x_inmm,y_inmm,z_inmm):
        txyz.append([t,x,y,z])

    return txyz

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

def cosmic_veto(not_accepted_candidates_mask, labels_cosmics, tyz, candidate_t,candidate_x,candidate_y, candidate_z, time_window, space_window):
    ### function for applying a veto on 39Ar candidates near cosmic tracks. Not validated or tested recently.
    #, tyz, candidate_t, time_window=300
        # loop through large clusters
        labels_cosmics = np.array(labels_cosmics)
        labels_cosmics = labels_cosmics[labels_cosmics != -1]
        track_lengths = []
        total_time_track = []
        print("total cosmic like clusters = ", np.size(np.unique(labels_cosmics)))
        for i, label in enumerate(np.unique(labels_cosmics)):
            packet_indices = np.where(labels_cosmics == label)[0]
            if np.size(packet_indices)!=0:
                packet_tyz = tyz[packet_indices]
                packet_t = packet_tyz[:,0]
                packet_y = packet_tyz[:,1]
                packet_z = packet_tyz[:,2]
                total_time = (np.max(packet_t) - np.min(packet_t))/1.6
                total_time_track.append(total_time)

                t_min, t_max = np.min(packet_t), np.max(packet_t)
                y_min, y_max = np.min(packet_y), np.max(packet_y)
                z_min, z_max = np.min(packet_z), np.max(packet_z)

                veto_check = np.zeros_like(candidate_t, dtype=bool)

                max_Delta_t_mm = 1000 # only apply cut for track-like events that aren't too long in time
                if total_time < max_Delta_t_mm:
                    #veto_check = (candidate_t > t_min - time_window) & (candidate_t < t_max + time_window) # & (track_length < track_cut)
                    
                    #veto_check = (candidate_t > t_min - time_window) & (candidate_t < t_max + time_window)\
                    #& (candidate_y > y_min - space_window) & (candidate_y < y_max + space_window) \
                    #    & (candidate_z > z_min - space_window) & (candidate_z < z_max + space_window) #& (track_length < track_cut)
                
                    veto_check = (candidate_t > t_min - time_window) & (candidate_t < t_max + time_window) \
                        | (candidate_y > 200) | (candidate_y < -200) | (candidate_x > 200) | (candidate_x < -200)

                not_accepted_candidates_mask = not_accepted_candidates_mask + veto_check
        
        accepted_candidates_mask = np.invert(not_accepted_candidates_mask)
        return accepted_candidates_mask

def build_charge_events(labels_noise_list,dataword,all_charge,txyz,v_ref,v_cm,v_ped,gain):
    ### Build charge events by adding up packet charge from individual DBSCAN clusters
    # Inputs: 
    #   labels_noise_list: list of noise labels from DBSCAN
    #   dataword: packet ADC counts
    #   all_charge: array to fill with total charge per cluster
    #   unique_ids: unique id for each pixel corresponding to the packets
    #   v_ref, v_cm, v_ped, gain: arrays providing pixel parameters for ADC->ke- conversion
    # Outputs:
    #   all_charge: filled with total cluster charge in ke-

    charge_cut = 3.0 # cut to make on hit charge in ke-
    charge = adcs_to_ke(dataword, v_ref,v_cm,v_ped,gain)
    charge_cut_mask = charge > charge_cut
    labels_noise = np.array(labels_noise_list)[charge_cut_mask]

    q_vals = np.bincount(labels_noise, weights=charge[charge_cut_mask])
    txyz = np.array(txyz)
    #print('length of txyz = ', len(txyz))

    # find midpoint of clustered hits and save array of all event information
    n_vals = np.bincount(labels_noise)

    t_vals = np.bincount(labels_noise, weights=txyz[:,0][charge_cut_mask])[n_vals != 0] # add up x values of hits in cluster then avg
    x_vals = np.bincount(labels_noise, weights=txyz[:,1][charge_cut_mask])[n_vals != 0]
    y_vals = np.bincount(labels_noise, weights=txyz[:,2][charge_cut_mask])[n_vals != 0]
    z_vals = np.bincount(labels_noise, weights=txyz[:,3][charge_cut_mask])[n_vals != 0]
    q_vals = q_vals[n_vals != 0]
    n_vals = n_vals[n_vals != 0] # get rid of n_vals that are 0, otherwise get divide by 0 later
    
    t_vals = t_vals[q_vals != 0]
    x_vals = x_vals[q_vals != 0]
    y_vals = y_vals[q_vals != 0]
    z_vals = z_vals[q_vals != 0]
    n_vals = n_vals[q_vals != 0]
    q_vals = q_vals[q_vals != 0]

    nqtxyz = np.zeros((len(n_vals[n_vals != 0]), 6))
    nqtxyz[:,0] = n_vals
    nqtxyz[:,1] = q_vals
    nqtxyz[:,2] = t_vals/n_vals # average for each event
    nqtxyz[:,3] = x_vals/n_vals
    nqtxyz[:,4] = y_vals/n_vals
    nqtxyz[:,5] = z_vals/n_vals
    return nqtxyz

def adcs_to_ke(adcs, v_ref, v_cm, v_ped, gain):
    ### converts adc counts to charge in ke-
    # Inputs:
    #   adcs: array of packet ADC counts
    #   v_ref, v_cm, v_ped, gain: array of pixel calibration parameters
    #   indices: array of indices
    # Outputs:
    #   array of charge in ke- 
    charge = (adcs.astype('float64')/float(fee.ADC_COUNTS)*(v_ref - v_cm)+v_cm-v_ped)/gain * 1e-3
    return charge

def pedestal_and_config(unique_ids, mc_assn):
    # function to open the pedestal and configuration files to get the dictionaries
    #   for the values of v_ped, v_cm, and v_ref for individual pixels. 
    #   Values are fixed in simulation but vary in data depending on pixel.
    # Inputs:
    #   unique_ids: 
    #       note: array containing a unique id for each pixel
    #       size: same as packets dataset (after selections)
    #   mc_assn:
    #       note: mc_truth information for simulation (None for data)
    # Returns:
    #   v_ped, v_cm, v_ref, gain arrays; size of packets dataset

    pedestal_file = 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
    config_file = 'evd_config_21-03-31_12-36-13.json'

    config_dict = defaultdict(lambda: dict(
        vref_mv=1300,
        vcm_mv=288
    ))
    pedestal_dict = defaultdict(lambda: dict(
        pedestal_mv=580
    ))

    # reading the data from the file
    with open(pedestal_file,'r') as infile:
        for key, value in json.load(infile).items():
            pedestal_dict[key] = value

    with open(config_file, 'r') as infile:
        for key, value in json.load(infile).items():
            config_dict[key] = value
  
    v_ped,v_cm,v_ref,gain = np.zeros_like(unique_ids,dtype='float64'),np.zeros_like(unique_ids,dtype='float64'),np.zeros_like(unique_ids,dtype='float64'),np.ones_like(unique_ids,dtype='float64')

    # make arrays with values for v_ped,v_cm,v_ref, and gain for ADC to ke- conversion 
    for i,id in enumerate(unique_ids):
        if not mc_assn:
            v_ped[i] = pedestal_dict[id]['pedestal_mv']
            v_cm[i] = config_dict[id]['vcm_mv']
            v_ref[i] = config_dict[id]['vref_mv']
            gain[i] = gain_data
        else:
            v_ped[i] = v_pedestal_sim
            v_cm[i] = v_cm_sim
            v_ref[i] = v_ref_sim
            gain[i] = gain_sim
    return v_ped, v_cm, v_ref, gain

def timestamp_corrector(packets, mc_assn):
    # Corrects larpix clock timestamps due to slightly different PACMAN clock frequencies 
    # (from module0_flow timestamp_corrector.py)
    ts = packets['timestamp'].astype('f8')

    if mc_assn == None:
        timestamp_cut = (packets['timestamp'] > 2e7) | (packets['timestamp'] < 1e6)
        ts = ts[np.invert(timestamp_cut)]
        packets = packets[np.invert(timestamp_cut)]

    #packet_type_0 = packets['packet_type'] == 0
    #if mc_assn: # optionally, cut data packets without mc_truth
    #    mc_assn = mc_assn[np.invert(timestamp_cut)]
    #    mc_assn_tracks = mc_assn['track_ids'][packet_type_0]
    #    packets = packets[mc_assn_tracks[:,0] != -1]
    #    ts_corr = ts_corr[mc_assn_tracks[:,0] != -1]

    if mc_assn == None:
        # only supports module-0
        correction1 = [-9.597, 4.0021e-6]
        correction2 = [-9.329, 1.1770e-6]
        mask_io1 = packets['io_group'] == 1
        mask_io2 = packets['io_group'] == 2
        ts[mask_io1] = (packets[mask_io1]['timestamp'].astype('f8') - correction1[0]) / (1. + correction1[1])
        ts[mask_io2] = (packets[mask_io2]['timestamp'].astype('f8') - correction2[0]) / (1. + correction2[1])

    # correct for rollovers
    rollover_ticks = 1e7
    #Create arrays to keep track of rollovers
    rollover_io1 = np.zeros(packets.shape[0], dtype = int)
    rollover_io2 = np.zeros(packets.shape[0], dtype = int)

    #Check for rollovers
    rollover_io1[(packets['io_group'] == 1) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)] = rollover_ticks
    rollover_io2[(packets['io_group'] == 2) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)] = rollover_ticks

    #Reset the rollover arrays
    rollover_io1 = np.cumsum(rollover_io1)
    rollover_io2 = np.cumsum(rollover_io2)
    
    #Apply the rollovers to ts
    ts[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] += rollover_io1[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] - rollover_ticks
    ts[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)] += rollover_io1[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)]
    ts[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] += rollover_io2[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] - rollover_ticks
    ts[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)] += rollover_io2[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)]
    
    packet_type_0 = packets['packet_type'] == 0
    ts = ts[packet_type_0]
    packets = packets[packet_type_0]
    
    sorted_idcs = np.argsort(ts)
    ts_corr_sorted = ts[sorted_idcs]
    packets_sorted = packets[sorted_idcs]
    return ts_corr_sorted, packets_sorted

def getPackets(file, sel_start, sel_end):
    #packets = packets[packets['valid_parity'] == 1]
    mc_assn=None
    try:
        mc_assn = file['mc_packets_assn']
    except:
        mc_assn=None
    
    # load packets and make selection
    packets = file['packets']
    if sel_end == -1:
        sel_end = len(packets)
    packets = packets[sel_start:sel_end]
    ts, packets = timestamp_corrector(packets, mc_assn)

    return ts, packets, mc_assn

def calibrations(packets, mc_assn):
     # unique id for each pixel, not to be confused with larnd-sim's pixel id
    unique_ids = ((((packets['io_group'].astype(int)) * 256
        + packets['io_channel'].astype(int)) * 256
        + packets['chip_id'].astype(int)) * 64 \
        + packets['channel_id'].astype(int)).astype(str)
    #### call v_ped ,etc here
    v_ped, v_cm, v_ref, gain = pedestal_and_config(unique_ids, mc_assn)
    return v_ped, v_cm, v_ref, gain
    
def analysis(file,pixel_xy,sel_start=0, sel_end=-1,use_veto=False,time_window=0,space_window=0,cut=False):
    ts, packets, mc_assn = getPackets(file, sel_start, sel_end)
    dataword = packets['dataword']
    # zip up y, z, and t values for clustering
    txyz = zip_pixel_tyz(packets,ts, pixel_xy)
    v_ped, v_cm, v_ref, gain = calibrations(packets, mc_assn)

    # apply cuts 
    if mc_assn == None and cut == True:
        txyz_array = np.array(txyz)
        z_array = txyz_array[:,3]
        x_array = txyz_array[:,1]
        y_array = txyz_array[:,2]
        ## cut the two noisy tiles
        #cut_io1 = (y_array > 0) & (y_array < 310) & (x_array < 0) & (x_array > -310) & (z_array > 0)
        #cut_io2 = (y_array < -310) & (y_array > -620) & (x_array > 0) & (x_array < 310) & (z_array < 0)
        ## cut only the noisy parts of the two noisy tiles
        cut_io1 = (y_array > 0) & (y_array < 310) & (x_array < -250) & (x_array > -310) & (z_array > 0)
        cut_io2 = (y_array < -310) & (y_array > -620) & (x_array > 250) & (x_array < 310) & (z_array < 0)
        ## cut two slices of tiles that are not noisy
        #cut_io1 = (y_array > 310) & (y_array < 620) & (x_array < -250) & (x_array > -310) & (z_array > 0)
        #cut_io2 = (y_array < 0) & (y_array > -310) & (x_array > 250) & (x_array < 310) & (z_array < 0)
        ## cut inner regions of tiles
        #cut_io1 = (x_array < -100) | (x_array > 100) & (z_array > 0)
        #cut_io2 = (x_array < -100) | (x_array > 100) & (z_array < 0)

        cut_total = np.invert(cut_io1 + cut_io2)
        txyz_array = txyz_array[cut_total]
        txyz = list(txyz_array)
        dataword = dataword[cut_total]
        v_ped = v_ped[cut_total]
        v_cm = v_cm[cut_total]
        v_ref = v_ref[cut_total]
        gain = gain[cut_total]

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
        
        accepted_candidates_mask = np.ones_like(candidate_t,dtype=bool)
        # loop through large clusters to do cosmics veto
        if use_veto and not mc_assn:
            not_accepted_candidates_mask = np.zeros_like(candidate_t,dtype=bool)
            accepted_candidates_mask = cosmic_veto(not_accepted_candidates_mask, labels_cosmics, np.array(txyz),candidate_t,candidate_x,candidate_y,candidate_z,time_window,space_window)
        
        # cluster packets that remain after tracks have been removed
        txyz_core, txyz_noise_2, txyz_noncore, db_noise = cluster_packets(eps_noise, min_samples_noise,txyz_noise)
        labels_noise = db_noise.labels_
        labels_noise_list = list(labels_noise)
        noise_samples_mask = db.labels_ == -1
    else:
        start = time.time()
        txyz_core, txyz_noise_2, txyz_noncore, db_noise = cluster_packets(eps_noise, min_samples_noise,txyz)
        end = time.time()
        labels_noise = db_noise.labels_
        labels_noise_list = list(labels_noise)
        txyz_noise = txyz
        #noise_samples_mask = db.labels_ == -1
    
    if mc_assn == None:
        labels_noise = np.array(labels_noise_list)[accepted_candidates_mask]
        dataword = np.array(dataword)[noise_samples_mask][accepted_candidates_mask]
        #unique_ids = unique_ids[noise_samples_mask][accepted_candidates_mask]
        v_ref = v_ref[noise_samples_mask][accepted_candidates_mask]
        v_cm = v_cm[noise_samples_mask][accepted_candidates_mask]
        v_ped = v_ped[noise_samples_mask][accepted_candidates_mask]
        gain = gain[noise_samples_mask][accepted_candidates_mask]
    else:
        dataword = np.array(dataword)
    
    all_charge_array = np.zeros_like(np.unique(labels_noise),dtype='float')
    
    # build charge events out of 39Ar candidates
    nqtxyz = build_charge_events(labels_noise,dataword,all_charge_array,txyz_noise,\
                v_ref=v_ref,v_cm=v_cm,v_ped=v_ped,gain=gain)

    return nqtxyz

def remove_tracks(file,pixel_xy,sel_start=0, sel_end=-1):
    ts, packets, mc_assn = getPackets(file, sel_start, sel_end)
    # zip up y, z, and t values for clustering
    txyz = zip_pixel_tyz(packets,ts, pixel_xy)
    # cluster packets to find track-like charge events
    txyz_core, txyz_noise, txyz_noncore, db = cluster_packets(eps_tracks, min_samples_tracks, txyz)
    noise = db.labels_ == -1
    packets = packets[noise]
    timestamp_cut = (packets['timestamp'] > 2e7) | (packets['timestamp'] < 1e6) & (packets['packet_type'] == 0)
    packets = packets[np.invert(timestamp_cut)]
    return packets

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
    #vcm_mv = 288.28125
    #vref_mv = 1300.78125
    vcm_mv = v_cm_data
    vref_mv = v_ref_data
    gain = gain_data

    if data_type == 'MC':
        vcm_mv = v_cm_sim
        vref_mv = v_ref_sim
        #vref_mv = vref_mv
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
    #R=1
    eV_per_e = 1
    if calibrate:
        #R = 0.6
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
