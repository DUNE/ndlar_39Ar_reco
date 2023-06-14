import h5py
import numpy as np
from adc64format import dtypes, ADC64Reader
from tqdm import tqdm
from consts import *

def ADC_drift(timestamps, adc, module):
    ## correct for the clock drift in the different ADCs
    # input: tai_ns
    ADC_drift_slope_0 = module.ADC_drift_slope_0
    ADC_drift_slope_1 = module.ADC_drift_slope_1
    clock_correction_factor = module.clock_correction_factor
    
    if adc == 0:
        slope = ADC_drift_slope_0
    elif adc == 1:
        slope = ADC_drift_slope_1
    elif adc > 1 or adc < 0:
        raise ValueError('adc must be either 0 or 1.')
        
    return timestamps.astype('f8')/(1.0 + slope) * clock_correction_factor
    
def into_array(events_file_0, events_file_1, batch_size, event_dtype, module, adc_sn_1, adc_sn_2):
    events = np.zeros((0,), dtype=event_dtype)
    # loop through events in batch
    for i in range(batch_size):
        if events_file_0['time'][i] is not None and events_file_1['time'][i] is not None:
            event_time_0 = events_file_0['time'][i]
            event_time_1 = events_file_1['time'][i]
            if event_time_0['tai_s'] == event_time_1['tai_s']:
                # get relevant info about event from each ADC
                event_tai_s = event_time_0[0]['tai_s']
                if module.detector == 'module-0':
                    event_tai_ns = (0.5*(ADC_drift(event_time_0[0]['tai_ns'] + event_time_0[0]['tai_s']*1e9, 0, module) + \
                    ADC_drift(event_time_1[0]['tai_ns'] + event_time_1[0]['tai_s']*1e9, 1, module))).astype('i8')
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
def read_light_files(module):
    #print('Loading light files with a batch size of ', module.batch_size, ' ...')
    input_light_filename_1 = module.adc_folder + module.input_light_filename_1
    input_light_filename_2 = module.adc_folder + module.input_light_filename_2
    nSec_start = module.nSec_start_light
    nSec_end = module.nSec_end_light
    adc_sn_1 = (module.input_light_filename_1).split('_')[0]
    adc_sn_2 = (module.input_light_filename_2).split('_')[0]
    
    go = True
    event_dtype = np.dtype([('tai_s', '<i4'), ('tai_ns', '<i8'), ('unix', '<i8'), ('channel_'+adc_sn_1, 'u1' , (module.nchannels_adc1,)),('channel_'+adc_sn_2, 'u1' , (module.nchannels_adc2,)), ('voltage_'+adc_sn_1,'<i2',(module.nchannels_adc1, module.light_time_steps)), ('voltage_'+adc_sn_2,'<i2',(module.nchannels_adc2,module.light_time_steps))])
    light_events_all = np.zeros((0,),dtype=event_dtype)
    # read through the LRS files to get the light triggers
    with ADC64Reader(input_light_filename_1, input_light_filename_2) as reader:
        current_sec = 0
        pbar = tqdm(total=nSec_end-nSec_start, desc=' Seconds processed in the light data:')
        while go == True:
            events = reader.next(batch_size)
            # get matched events between multiple files
            events_file_0, events_file_1 = events
            # convert to numpy arrays for easier access
            light_events = into_array(events_file_0, events_file_1, batch_size, event_dtype, module, \
                                     adc_sn_1, adc_sn_2)
            # combine each batch together
            if current_sec >= nSec_start and current_sec <= nSec_end:
                light_events_all = np.concatenate((light_events_all,light_events))
            
            set_of_tai_s = np.unique(light_events['tai_s'])
            
            if module.detector == 'module-0':
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
    
    
    
    
    
    
    

        
