import os
import h5py
import numpy as np
from tqdm import tqdm

def combine_h5_files(folder_path, output_file):
    # List all .h5 files in the given directory
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError("No .h5 files found in the specified folder.")
    
    combined_data = None
    dtype = None
    
    for h5_file in tqdm(h5_files):
        file_path = os.path.join(folder_path, h5_file)
        with h5py.File(file_path, 'r') as f:
            if 'events' not in f:
                raise KeyError(f"No 'events' dataset found in file {h5_file}")
            events_data = f['events'][:]
            
            if combined_data is None:
                combined_data = events_data
                dtype = events_data.dtype
            else:
                combined_data = np.concatenate((combined_data, events_data), axis=0)
    
    # Write combined data to the output file
    output_path = os.path.join(folder_path, output_file)
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('events', data=combined_data)
    
    # Remove 'samples' dtype and create new file
    dtype_no_samples = np.dtype([(name, dtype[name]) for name in dtype.names if name != 'samples'])
    combined_data_no_samples = np.zeros(combined_data.shape, dtype=dtype_no_samples)
    
    for name in dtype_no_samples.names:
        combined_data_no_samples[name] = combined_data[name]
    
    output_file_no_waveforms = output_file.replace('.h5', '_NoWaveforms.h5')
    output_path_no_waveforms = os.path.join(folder_path, output_file_no_waveforms)
    with h5py.File(output_path_no_waveforms, 'w') as f_out_no_waveforms:
        f_out_no_waveforms.create_dataset('events', data=combined_data_no_samples)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Combine .h5 files into a single .h5 file and create another without waveforms.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing .h5 files.')
    parser.add_argument('output_file', type=str, help='Name of the output .h5 file.')

    args = parser.parse_args()

    combine_h5_files(args.folder_path, args.output_file)