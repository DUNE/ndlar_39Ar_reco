import h5py
import numpy as np
from adc64format import dtypes, ADC64Reader
from tqdm import tqdm
from consts import *

def into_array(events_file, batch_size, event_dtype):
    length_arr = len(events_file['time'])
    events_ADC = np.zeros(length_arr, dtype=event_dtype)
    j=0
    for i in range(batch_size):
        if events_file['time'][i] is not None:
            event_time = events_file['time'][i]
            event_tai_s = event_time[0]['tai_s']
            event_tai_ns = event_time[0]['tai_ns']

            event_data = events_file['data'][i]
            event_channels = event_data['channel']
            event_voltages = event_data['voltage']
            events_ADC[j]['tai_s'] = event_tai_s
            events_ADC[j]['tai_ns'] = event_tai_ns
            events_ADC[j]['channel'] = np.array(event_channels, dtype='u1')
            events_ADC[j]['voltage'] = event_voltages
            
            event_header = events_file['header'][i]['unix']
            events_ADC[j]['unix'] = event_header*1e-3
            j+=1
    return events_ADC

# go through relevant seconds in .data files for LRS
def read_light_files(nSec_start, nSec_end):
    go = True
    event_dtype = np.dtype([('tai_s', '<u4'), ('tai_ns', '<u8'),('unix', '<u8'), ('channel', 'u1' , (58,)), ('voltage','<i2',(58,256))])
    events_314C_all = np.zeros((0,),dtype=event_dtype)
    events_54BD_all = np.zeros((0,),dtype=event_dtype)
    # read through the LRS files to get the light triggers
    with ADC64Reader(adc_file_1, adc_file_2) as reader:
        current_sec_314C = 0
        current_sec_54BD = 0
        pbar_314C = tqdm(total=nSec_end-nSec_start, desc='Seconds processed in the 314C light data:')
        pbar_54BD = tqdm(total=nSec_end-nSec_start, desc='Seconds processed in the 54BD light data:')
        while go == True:
            events = reader.next(batch_size)
            # get matched events between multiple files
            events_file0, events_file1 = events
            # convert to numpy arrays for easier access
            events_314C = into_array(events_file0, batch_size, event_dtype)
            events_54BD = into_array(events_file1, batch_size, event_dtype)

            # combine each batch together
            if current_sec_314C >= nSec_start and current_sec_314C <= nSec_end:
                events_314C_all = np.concatenate((events_314C_all,events_314C))
            if current_sec_54BD >= nSec_start and current_sec_54BD <= nSec_end:
                events_54BD_all = np.concatenate((events_54BD_all,events_54BD))

            set_of_tai_s_314C = np.unique(events_314C['tai_s'])
            set_of_tai_s_54BD = np.unique(events_54BD['tai_s'])
            # update pbar and iterate current_sec ~when a PPS pulse happens
            if len(set_of_tai_s_314C) > 1 and np.max(set_of_tai_s_314C) == 1:
                if current_sec_314C >= nSec_start and current_sec_314C <= nSec_end:
                    pbar_314C.update(1)
                current_sec_314C += 1
            if len(set_of_tai_s_54BD) > 1 and np.max(set_of_tai_s_54BD) == 1:
                if current_sec_54BD >= nSec_start and current_sec_54BD <= nSec_end:
                    pbar_54BD.update(1)
                current_sec_54BD += 1
            if current_sec_314C > nSec_end and current_sec_54BD > nSec_end:
                go = False
        pbar_314C.close()
        pbar_54BD.close()
        print('Finished reading light data.')
    return events_314C_all, events_54BD_all

# match the light triggers to packet type 7s in the packets dataset
def match(events, PPS_charge, unix_charge, label):
    ## INPUT
    # events: events array produced by read_light_files()
    # PPS_charge: time since last PPS of each packet type 7
    # unix_charge: unix time of each packet type 7

    # loop through light triggers and match to packet type 7s
    matched_triggers_unix = np.zeros_like(unix_charge, dtype=bool)
    matched_triggers_PPS = np.zeros_like(unix_charge, dtype=bool)
    indices = np.ones(len(events))*-1 # indices in the pkt type 7 unix/PPS arrays

    for i in tqdm(range(len(events)), desc=' Matching ' + label + ' light triggers to packets: '):
        light_unix = events[i]['unix']
        PPS_light = events[i]['tai_ns']
        #print(light_unix, ' ', np.unique(unix_charge)[2:5])
        #print(PPS_light, ' ', np.max(PPS_charge))
        unix_matched_trigger_mask = (unix_charge < light_unix + matching_tolerance_unix)\
            & (unix_charge > light_unix - matching_tolerance_unix)
        PPS_matched_trigger_mask = (PPS_charge < PPS_light + matching_tolerance_PPS) \
            & (PPS_charge > PPS_charge - matching_tolerance_PPS)
        unix_PPS_matched_trigger_mask = (unix_matched_trigger_mask) & (PPS_matched_trigger_mask)
        #print(np.sum(unix_PPS_matched_trigger_mask))
        trigger_index = np.where(unix_PPS_matched_trigger_mask)[0]
        #print(trigger_index)
        if len(trigger_index) == 1:
            indices[i] = trigger_index 
        matched_triggers_unix += unix_matched_trigger_mask
        matched_triggers_PPS += PPS_matched_trigger_mask
        matched_triggers = (matched_triggers_unix) & (matched_triggers_PPS)
    print('Total matched triggers = ', np.sum(matched_triggers), ' out of ', len(unix_charge), ' total triggers.')
    return matched_triggers, indices
    
#def find_matched_charge_events(events, indices, PPS_charge):
#    # find charge events near light triggers
#    events[]
        
        
