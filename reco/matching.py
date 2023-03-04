import numpy as np
import h5py
from consts import *
from tqdm import tqdm

# match the light triggers to packet type 7s in the packets dataset
def match_light_to_ext_trigger(events, PPS_charge, unix_charge, matching_tolerance_unix, matching_tolerance_PPS):
    ## INPUT
    # events: events array produced by read_light_files()
    # PPS_charge: time since last PPS of each packet type 7
    # unix_charge: unix time of each packet type 7

    # loop through light triggers and match to packet type 7s
    matched_triggers_unix = np.zeros_like(unix_charge, dtype=bool)
    matched_triggers_PPS = np.zeros_like(unix_charge, dtype=bool)
    matched_triggers = np.zeros_like(unix_charge, dtype=bool)
    indices_in_ext_triggers = np.ones(len(events), dtype='i8')*-1 # indices in the pkt type 7 unix/PPS arrays
    
    for i in tqdm(range(len(events)), desc=' Matching light triggers to packets: '):
        light_unix = events[i]['unix']
        PPS_light = events[i]['tai_ns']

        unix_matched_trigger_mask = np.abs(unix_charge - light_unix) <= matching_tolerance_unix
        PPS_matched_trigger_mask = np.abs(PPS_charge.astype('i8') - PPS_light) <= matching_tolerance_PPS
        unix_PPS_matched_trigger_mask = (unix_matched_trigger_mask) & (PPS_matched_trigger_mask)
        trigger_index = np.where(unix_PPS_matched_trigger_mask)[0] # points to element in PPS_pt7
        
        if len(trigger_index) == 2: # only accept if there's two ext triggers, one for each PACMAN
            # usually two ext triggers per light trig, one for each pacman
            # but assuming now that the time difference b/w them is very small <1usec
            indices_in_ext_triggers[i] = np.min(trigger_index) 
                                         
        matched_triggers_unix += unix_matched_trigger_mask
        matched_triggers_PPS += PPS_matched_trigger_mask
        matched_triggers += (unix_matched_trigger_mask) & (PPS_matched_trigger_mask)
        
    print('Total matched triggers = ', np.sum(matched_triggers), ' out of ', len(unix_charge), ' total triggers.')
    print('Total matched triggers based on unix only = ', np.sum(matched_triggers_unix), ' out of ', len(unix_charge), ' total triggers.')
    print('Total matched triggers based on PPS only = ', np.sum(matched_triggers_PPS), ' out of ', len(unix_charge), ' total triggers.')
    return indices_in_ext_triggers
    
def match_light_to_charge(light_events, charge_events, PPS_ext_triggers, unix_ext_triggers, cluster_type):
    ### match light triggers to charge events
    # cluster type: 0 for small clusters, 1 for large clusters
    if cluster_type == 0:
        charge_event_ns = charge_events['t']
    elif cluster_type == 1:
        charge_event_ns_min = charge_events['t_min']
        charge_event_ns_max = charge_events['t_max']
    charge_event_unix = charge_events['unix']
    num_matched_0 = 0
    num_matched_1 = 0
    num_matched_2 = 0
    num_matched_3 = 0
    num_matched_more = 0
    
    PPS_window = int(drift_distance / v_drift * 1e3)
    unix_window = 1 # s
    matched_light_index = 0
    indices_in_light_events = []
    # loop through light events and check for charge event within a drift window
    for i in tqdm(range(len(light_events)), desc = ' Matching light events to charge events: '):
        light_event = light_events[i]
        PPS_ext_trigger = PPS_ext_triggers[i]
        unix_ext_trigger = unix_ext_triggers[i]
        if cluster_type == 0:
            matched_events_PPS = (charge_event_ns > PPS_ext_trigger) & (charge_event_ns < PPS_ext_trigger + PPS_window)
            matched_events_unix = (charge_event_unix > unix_ext_trigger-unix_window) & (charge_event_unix < unix_ext_trigger + unix_window)
        elif cluster_type == 1:
            matched_events_PPS = (charge_event_ns_min > PPS_ext_trigger) & (charge_event_ns_max < PPS_ext_trigger + PPS_window)
            matched_events_unix = (charge_event_unix > unix_ext_trigger-unix_window) & (charge_event_unix < unix_ext_trigger + unix_window)
        matched_events = (matched_events_PPS) & (matched_events_unix)
        matched_events_indices = np.where(matched_events)[0]
        if len(matched_events_indices) > 0:
            indices_in_light_events.append(i)
            for index in matched_events_indices:
                charge_events[index]['matched'] = 1
                charge_events[index]['light_index'] = matched_light_index
            matched_light_index += 1
        if len(matched_events_indices) == 0:
            num_matched_0 += 1
        elif len(matched_events_indices) == 1:
            num_matched_1 += 1
        elif len(matched_events_indices) == 2:
            num_matched_2 += 1
        elif len(matched_events_indices) == 3:
            num_matched_3 += 1
        elif len(matched_events_indices) > 3:
            num_matched_more += 1
    results_light_events = light_events[np.array(indices_in_light_events)]
    print('Number of times light is matched to 0, 1, 2, 3, or >3 different charge events: ', num_matched_0,' ', num_matched_1, ' ',num_matched_2,' ', num_matched_3,' ', num_matched_more)
    return charge_events, results_light_events
        