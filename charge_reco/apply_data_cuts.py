#!/usr/bin/env python
"""
Script for applying a cut to only include small clusters near the photon detectors.
As input requires a charge-light matched file with charge clusters and light waveforms.
The goal of these cuts are to create a high purity sample.

Output: 
     - file with the same format as the input but only includes events after cuts. 
     - various plots
"""

import fire
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_data_cuts(input_filepath, use_disabled_channels_cut=True):
    
    # check that the input file exists
    if not os.path.exists(input_filepath):
        raise Exception(f'File {input_filepath} does not exist.')
    else:
        print('Opening file: ', input_filepath)
    
    # open input file 
    f_in = h5py.File(input_filepath, 'r')
    
    # check that the relevant datasets are in the input file, if not raise error
    try:
        f_in['clusters']
    except:
        raise Exception("'clusters' dataset not found in input file.")
    try:
        f_in['light_events']
    except:
        raise Exception("'light_events' dataset not found in input file.")
    
    f_in.close()
if __name__ == "__main__":
    fire.Fire(apply_data_cuts)

