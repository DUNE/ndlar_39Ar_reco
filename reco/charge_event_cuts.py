import numpy as np
from consts import *
from tqdm import tqdm
from numba import njit, prange
import time

#@njit(parallel=True)
def find_spaced_out_elements(PPS_arr, unix_arr, PPS_window, unix_window):
    ## applies a cut on arr to exclude elements that are too close to any other elements
    ## can be used to only include charge events that occur >= a drift window from other events
    ## x: value of cut, nominally a drift window in ns
    ## arr: array of values to cut, nominally a timestamp array in ns
    #PPS_arr = PPS_arr.astype('f8') # the np.inf trick breaks with ints
    #unix_arr = unix_arr.astype('f8')
    events_mask = np.ones_like(PPS_arr, dtype='bool')
    for i in tqdm(range(len(PPS_arr)), desc=' Doing drift window cut on charge events: '):
    #for i in range(len(PPS_arr)):
        PPS_value = PPS_arr[i]
        unix_value = unix_arr[i]
        PPS_diff = np.abs(PPS_value - PPS_arr)
        unix_diff = np.abs(unix_value - unix_arr)
        # to avoid self-comparisons
        PPS_diff[i] = np.inf
        unix_diff[i] = np.inf
        # find events that occur outside one of the windows. These are events we keep.
        current_mask = (PPS_diff > PPS_window) | (unix_diff > unix_window)
        # compare to overall mask to make sure an event is rejected if it is a window away from any other event
        events_mask = (events_mask) & (current_mask)
    
    return events_mask
    
def charge_event_drift_window_cut(events):
    PPS_window = drift_distance / v_drift * 1e3 # ns
    unix_window = 1 # second
    PPS_timestamps = events['t'].astype('f8')
    unix_timestamps = events['unix'].astype('f8')
    cut_mask = find_spaced_out_elements(PPS_timestamps, unix_timestamps, PPS_window, unix_window)
    return cut_mask
    
def pixel_plane_cut(events):
    # apply cut to charge events based on pixel plane coordinates
    x_array = events['x']
    y_array = events['y']
    z_array = events['z']
    # cut only the noisy parts of the two noisy tiles (module-0)
    cut_io1 = (y_array > 0) & (y_array < 310) & (x_array < -250) & (x_array > -310) & (z_array > 0)
    cut_io2 = (y_array < -310) & (y_array > -620) & (x_array > 250) & (x_array < 310) & (z_array < 0)
    cut_total_inverted = np.invert(cut_io1 + cut_io2)
    return cut_total_inverted
    
def all_charge_event_cuts(events):
    # do all charge event cuts here for organization purposes
    if use_pixel_plane_cut:
        print(' ')
        print('Total charge events before pixel plane cut = ', len(events))
        pixel_plane_cut_mask = pixel_plane_cut(events)
        events = events[pixel_plane_cut_mask]
        print('Total charge events after pixel plane cut = ', len(events))
        print(' ')
    if use_charge_event_drift_window_cut:
        print('Total charge events before drift window cut = ', len(events))
        start = time.time()
        drift_window_cut_mask = charge_event_drift_window_cut(events)
        end = time.time()
        print('time to do drift window cut = ', end-start)
        events = events[drift_window_cut_mask]
        print('Total charge events after drift window cut = ', len(events))
    
    return events