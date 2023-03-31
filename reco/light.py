import h5py
import numpy as np
from adc64format import dtypes, ADC64Reader
from tqdm import tqdm
from consts import *

def ADC_drift(timestamps, adc, detector):
    ## correct for the clock drift in the different ADCs
    # input: tai_ns
    if detector == 'module-0':
        slope_0 = -1.18e-7 # sn 175780172 (#314C)
        slope_1 = 1.18e-7 # sn 175854781 (#54BD)
    elif detector == 'module-3':
        slope_0 = 0
        slope_1 = 0
    clock_correction_factor = 0.625
    
    if adc == 0:
        slope = slope_0
    elif adc == 1:
        slope = slope_1
    elif adc > 1 or adc < 0:
        raise ValueError('adc must be either 0 or 1.')
        
    return timestamps.astype('f8')/(1.0 + slope) * clock_correction_factor
    
def into_array(events_file_0, events_file_1, batch_size, event_dtype, detector,adc_sn_1, adc_sn_2):
    events = np.zeros((0,), dtype=event_dtype)
    
    # loop through events in batch
    for i in range(batch_size):
        if events_file_0['time'][i] is not None and events_file_1['time'][i] is not None:
            event_time_0 = events_file_0['time'][i]
            event_time_1 = events_file_1['time'][i]
            if event_time_0['tai_s'] == event_time_1['tai_s']:
                # get relevant info about event from each ADC
                event_tai_s = event_time_0[0]['tai_s']
                if detector == 'module-0':
                    event_tai_ns = (0.5*(ADC_drift(event_time_0[0]['tai_ns'] + event_time_0[0]['tai_s']*1e9, 0, detector) + \
                    ADC_drift(event_time_1[0]['tai_ns'] + event_time_1[0]['tai_s']*1e9, 1, detector))).astype('i8')
                else:
                    event_tai_ns = (0.5*(event_time_0[0]['tai_ns'] + event_time_1[0]['tai_ns'])).astype('i8')
                    
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
                event['channel_'+adc_sn_1] = np.array(event_channels_0, dtype='u1')
                event['channel_'+adc_sn_2] = np.array(event_channels_1, dtype='u1')
                event['voltage_'+adc_sn_1] = event_voltages_0
                event['voltage_'+adc_sn_2] = event_voltages_1
                event_header = events_file_0['header'][i]['unix'].astype('i8')
                event['unix'] = event_header*1e-3
                events = np.concatenate((events, event))
                
    return events

# go through relevant seconds in .data files for LRS
def read_light_files(adc_file_1, adc_file_2, nSec_start, nSec_end, detector, adc_steps, nchannels_adc1, nchannels_adc2, adc_sn_1,adc_sn_2):
    go = True
    event_dtype = np.dtype([('tai_s', '<i4'), ('tai_ns', '<i8'), ('unix', '<i8'), ('channel_'+adc_sn_1, 'u1' , (nchannels_adc1,)),('channel_'+adc_sn_2, 'u1' , (nchannels_adc2,)), ('voltage_'+adc_sn_1,'<i2',(nchannels_adc1,adc_steps)), ('voltage_'+adc_sn_2,'<i2',(nchannels_adc2,adc_steps)), ('light_unique_id', '<i4')])
    light_events_all = np.zeros((0,),dtype=event_dtype)
    # read through the LRS files to get the light triggers
    with ADC64Reader(adc_file_1, adc_file_2) as reader:
        current_sec = 0
        pbar = tqdm(total=nSec_end-nSec_start, desc=' Seconds processed in the light data:')
        while go == True:
            events = reader.next(batch_size)
            # get matched events between multiple files
            events_file_0, events_file_1 = events
            # convert to numpy arrays for easier access
            light_events = into_array(events_file_0, events_file_1, batch_size, event_dtype, detector, \
                                     adc_sn_1, adc_sn_2)
            # combine each batch together
            if current_sec >= nSec_start and current_sec <= nSec_end:
                light_events_all = np.concatenate((light_events_all,light_events))
            
            set_of_tai_s = np.unique(light_events['tai_s'])
            
            if detector == 'module-0':
                if len(set_of_tai_s) > 1 and np.max(set_of_tai_s) == 1:
                    if current_sec >= nSec_start and current_sec <= nSec_end:
                        pbar.update(1)
                    current_sec += 1
                if current_sec > nSec_end:
                    go = False
            else:
                if len(set_of_tai_s) > 1:
                    if current_sec >= nSec_start and current_sec <= nSec_end:
                        pbar.update(1)
                    current_sec += 1
                if current_sec > nSec_end:
                    go = False
        pbar.close()
        print('Finished reading light data.')
    return light_events_all
    
    
    
    
    
    
    

        
