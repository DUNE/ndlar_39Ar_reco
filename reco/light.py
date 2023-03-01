import h5py
import numpy as np
from adc64format import dtypes, ADC64Reader
from tqdm import tqdm
from consts import *

def ADC_drift(timestamps, adc):
    ## correct for the clock drift in the different ADCs
    # input: tai_ns
    slope_0 = -1.18e-7 # sn 175780172 (#314C)
    slope_1 = 1.18e-7 # sn 175854781 (#54BD)
    clock_correction_factor = 0.625
    slope = 0
    if adc == 0:
        slope = slope_0
    elif adc == 1:
        slope = slope_1
    elif adc > 1 or adc < 0:
        raise ValueError('adc must be either 0 or 1.')
        
    return timestamps.astype('f8')/(1.0 + slope) * clock_correction_factor
    
def into_array(events_file_0, events_file_1, batch_size, event_dtype):
    events = np.zeros((0,), dtype=event_dtype)
    
    for i in range(batch_size):
        if events_file_0['time'][i] is not None and events_file_1['time'][i] is not None:
            event_time_0 = events_file_0['time'][i]
            event_time_1 = events_file_1['time'][i]
            if event_time_0['tai_s'] == event_time_1['tai_s']:
                # get relevant info about event from each ADC
                event_tai_s = event_time_0[0]['tai_s']
                event_tai_ns = (0.5*(ADC_drift(event_time_0[0]['tai_ns'] + event_time_0[0]['tai_s']*1e9, 0) + \
                    ADC_drift(event_time_1[0]['tai_ns'] + event_time_1[0]['tai_s']*1e9, 1))).astype('i8')
                event_data_0 = events_file_0['data'][i]
                event_data_1 = events_file_1['data'][i]
                event_channels_0 = event_data_0['channel']
                event_channels_1 = event_data_1['channel']
                event_voltages_0 = event_data_0['voltage']
                event_voltages_1 = event_data_1['voltage']
                
                # save event info to array
                event = np.zeros((1,), dtype=event_dtype)
                event['tai_s'] = event_tai_s
                event['tai_ns'] = event_tai_ns
                event['channel_314C'] = np.array(event_channels_0, dtype='u1')
                event['channel_54BD'] = np.array(event_channels_1, dtype='u1')
                event['voltage_314C'] = event_voltages_0
                event['voltage_54BD'] = event_voltages_1
                event_header = events_file_0['header'][i]['unix'].astype('i8')
                event['unix'] = event_header*1e-3
                events = np.concatenate((events, event))
                
    return events

# go through relevant seconds in .data files for LRS
def read_light_files(nSec_start, nSec_end):
    go = True
    #event_dtype = np.dtype([('tai_s', '<i4'), ('tai_ns', '<i8'),('unix', '<i8'), ('channel', 'u1' , (58,)), ('voltage','<i2',(58,256))])
    event_dtype = np.dtype([('tai_s', '<i4'), ('tai_ns', '<i8'), ('unix', '<i8'), ('channel_314C', 'u1' , (58,)),('channel_54BD', 'u1' , (58,)), ('voltage_314C','<i2',(58,256)), ('voltage_54BD','<i2',(58,256))])
    light_events_all = np.zeros((0,),dtype=event_dtype)
    # read through the LRS files to get the light triggers
    with ADC64Reader(adc_file_1, adc_file_2) as reader:
        current_sec = 0
        pbar = tqdm(total=nSec_end-nSec_start, desc='Seconds processed in the light data:')
        while go == True:
            events = reader.next(batch_size)
            # get matched events between multiple files
            events_file_0, events_file_1 = events
            # convert to numpy arrays for easier access
            light_events = into_array(events_file_0, events_file_1, batch_size, event_dtype)
            # combine each batch together
            if current_sec >= nSec_start and current_sec <= nSec_end:
                light_events_all = np.concatenate((light_events_all,light_events))
            
            set_of_tai_s = np.unique(light_events['tai_s'])
            
            if len(set_of_tai_s) > 1 and np.max(set_of_tai_s) == 1:
                if current_sec >= nSec_start and current_sec <= nSec_end:
                    pbar.update(1)
                current_sec += 1
            if current_sec > nSec_end:
                go = False
        #pbar_314C.close()
        #pbar_54BD.close()
        pbar.close()
        print('Finished reading light data.')
    #return events_314C_all, events_54BD_all
    return light_events_all

# match the light triggers to packet type 7s in the packets dataset
def match_to_ext_trigger(events, PPS_charge, unix_charge):
    ## INPUT
    # events: events array produced by read_light_files()
    # PPS_charge: time since last PPS of each packet type 7
    # unix_charge: unix time of each packet type 7

    # loop through light triggers and match to packet type 7s
    matched_triggers_unix = np.zeros_like(unix_charge, dtype=bool)
    matched_triggers_PPS = np.zeros_like(unix_charge, dtype=bool)
    matched_triggers = np.zeros_like(unix_charge, dtype=bool)
    indices_in_ext_triggers = np.ones(len(events), dtype='i8')*-1 # indices in the pkt type 7 unix/PPS arrays
    
    #for i in tqdm(range(len(events)), desc=' Matching ' + label + ' light triggers to packets: '):
    for i in tqdm(range(len(events)), desc=' Matching light triggers to packets: '):
        light_unix = events[i]['unix']
        PPS_light = events[i]['tai_ns']

        unix_matched_trigger_mask = np.abs(unix_charge - light_unix) <= matching_tolerance_unix
        PPS_matched_trigger_mask = np.abs(PPS_charge.astype('i8') - PPS_light) <= matching_tolerance_PPS/0.625
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
    return matched_triggers, indices_in_ext_triggers
    
def match_light_to_charge(light_events, charge_events, PPS_ext_triggers, unix_ext_triggers):
    charge_event_ns = charge_events['t']
    charge_event_unix = charge_events['unix']
    num_matched_0 = 0
    num_matched_1 = 0
    num_matched_2 = 0
    num_matched_3 = 0
    num_matched_more = 0
    print('charge event ns = ', charge_event_ns[0:50])
    print('charge event ns max = ', np.max(charge_event_ns))
    print('PPS_ext_triggers = ', PPS_ext_triggers[0:50])
    PPS_window = int(drift_distance / v_drift * 1e3)
    unix_window = 1 # s
    print('PPS_window = ', PPS_window)
    # loop through light events and check for charge event within a drift window
    for i in tqdm(range(len(light_events)), desc = ' Matching light events to charge events: '):
        light_event = light_events[i]
        PPS_ext_trigger = PPS_ext_triggers[i]
        unix_ext_trigger = unix_ext_triggers[i]
        
        matched_events_PPS = (charge_event_ns > PPS_ext_trigger) & (charge_event_ns < PPS_ext_trigger + PPS_window)
        matched_events_unix = (charge_event_unix > unix_ext_trigger-unix_window) & (charge_event_unix < unix_ext_trigger + unix_window)
        matched_events = (matched_events_PPS) & (matched_events_unix)
        matched_events_indices = np.where(matched_events)[0]
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
    print('Number of 0, 1, 2, 3, or >3 matches: ', num_matched_0,' ', num_matched_1, ' ',num_matched_2,' ', num_matched_3,' ', num_matched_more)
        
    
    
    
    
    
    
    

        
