#!/usr/bin/env python
"""
Command-line interface to LArNDLE
"""
import reco_fxns
import h5py
import fire
import time

def run_reconstruction(input_packets_filename, selection_start, selection_end):

    dict_path = 'layout/multi_tile_layout-2.3.16.pkl'
    pixel_xy = reco_fxns.load_geom_dict(dict_path)
    
    print('Opening packets file: ', input_packets_filename)
    print('Selecting packets ', selection_start, ' to ', selection_end)
    
    if input_packets_filename.split('.')[-1] != 'h5':
        raise Exception('Input file must be an h5 file.')
        
    f_packets = h5py.File(input_packets_filename)
    
    try:
        f_packets['packets']
    except:
        raise KeyError('Packets not found in ' + input_packets_filename)
    
    analysis_start = time.time()
    # outputs nqtxyz: nhit, charge, time since file start, x (mm) of hits, y (mm) of hits, z (mm) of hits; for events
    results = reco_fxns.analysis(f_packets,pixel_xy,sel_start=selection_start,sel_end=selection_end,cut=False)
    
    packets_filename_base = input_packets_filename.split('.h5')[0]
    output_events_filename = packets_filename_base + '_events.h5'
    print('Saving events to ', output_events_filename)
    with h5py.File(output_events_filename, 'w') as f:
        dset = f.create_dataset('small_clusters', data=results, dtype=results.dtype)
    
    analysis_end = time.time()
    print('Time to do analysis = ', analysis_end-analysis_start)
    print('Length of results = ', len(results))

if __name__ == "__main__":
    fire.Fire(run_reconstruction)