#!/usr/bin/env python
"""
Command-line interface to LArNDLE
"""
import reco_fxns
import h5py
import fire
import time

def run_reconstruction(input_packets_filename, selection_start, selection_end):

    dict_path = 'multi_tile_layout-2.3.16.pkl'
    pixel_xy = reco_fxns.load_geom_dict(dict_path)
    
    print('Running reco')
    print('Opening packets file: ', input_packets_filename)
    print('Selecting packets ', selection_start, ' to ', selection_end)
    
    f_packets = h5py.File(input_packets_filename)
    analysis_start = time.time()
    # outputs nqtxyz: nhit, charge, time since file start, x (mm) of hits, y (mm) of hits, z (mm) of hits; for events
    results = reco_fxns.analysis(f_packets,pixel_xy,sel_start=selection_start,sel_end=selection_end,cut=False)
    analysis_end = time.time()
    print('Time to do analysis = ', analysis_end-analysis_start)
    print('Length of results = ', len(results))

if __name__ == "__main__":
    fire.Fire(run_reconstruction)