### Overview
This code is tailored to reconstructing energy deposits from radiologicals in the ND-LAr prototypes. It uses DBSCAN to find clusters of charge hits. Since the code finds clusters of various sizes, one can make a selection on the output file to find small clusters corresponding to radiological deposits. 

Various reconstruction related scripts can be found in the `reco` directory. Useful shell scripts for running the reconstruction on many data files can be found in the `util` directory.

If light data is available, there is a script for matching the charge clusters to light triggers. This provides a t0 for the clusters that is used to calculate the drift coordinate. The way this works is for every light trigger, the code finds any clusters with timestamps within an asymmetric window around the charge cluster timestamps. While this accomplishes the task of matching light to charge, the purity of such a sample is generally poor (20-40% roughly). A cut is made to clusters that occur in close proximity to light detectors that have a summed waveform above a set threshold, which yields a much higher purity sample (>90%). 

### Setting up the code
To setup the code, run:
```bash
git clone https://github.com/sam-fogarty/ndlar_39Ar_reco.git
cd ndlar_39Ar_reco
pip install .
```

Here is an example of running the reconstruction on the commandline. First `cd` to the `charge_reco` directory, then:
```python
python3 charge_clustering.py module0 /path/to/input/packet/h5/file /path/to/output/h5/file
```
`module0` is an option inside in the ModuleConfig class inside `input_config.py`. It contains detector-specific configuration parameters and files. Make sure to use the correct detector configuration for the chosen input file. The input is a packetized h5 file containing the charge data. The output must be an h5 file. The reconstruction is done in batches (configurable) and a progress bar is shown. 

There are some optional input parameters:
- save_hits (**bool**): True to save hits in output file, False to not.
- match_to_ext_trig (**bool**): True to match clusters to external triggers. This is the old matching, so may go away eventually.
- pedestal_file (**str**): File path to a pedestal h5 file if you want to use the pedestal file directly instead of making the json file (finds channel by channel pedestals).
- v_cm_dac (**int**): v_cm_dac value used for the particular self-trigger run
- v_ref_dac (**int**): v_ref_dac value used for the particular self-trigger run

### Output Format
The output file contains the following datasets:

`clusters` : Note: Charge clusters found with DBSCAN 
 - id (**int**): unique cluster index
 - nhit (**int**): number of hits in cluster
 - q (**float**): total charge in mV in cluster (not corrected for electron lifetime or recombination)
 - io_group (**int**): io_group corresponding to cluster
 - t_max, t_mid, t_min (**int**): maximum/average/minimum PPS timestamp of cluster
 - t0 (**float**): matched external trigger PPS timestamp (-1 if not match)
 - x_max, x_mid, x_min (**float**): maximum/average/minimum pixel x position in mm
 - y_max, y_mid, y_min (**float**): maximum/average/minimum pixel y position in mm
 - z_anode (**float**): position of anode plane that detected this cluster in mm
 - z_drift_max, z_drift_mid, z_drift_min (**float**): maximum/average/minimum drift coordinate in mm
 - unix (**int**): unix timestamp of cluster
 - ext_trig_index (**int**): index of matched ext. trig in external trigger dataset (only used if matching to ext triggers is enabled)
 - light_trig_index (**int**): index of matched light trigger in `light_events` dataset

`hits` (Optional): Note: Charge hits
 - q (**float**): charge in mV
 - io_group (**int**): io_group corresponding to hit
 - unique_id (**int**): unique id of pixel
 - t (**int**): PPS timestamp of hit
 - x (**float**): x position of hit in mm
 - y (**float**): y position of hit in mm
 - z_anode (**float**): z position of hit in mm
 - z_drift (**float**): drift coordinate of hit in mm
 - unix (**int**): unix timestamp of hit
 - cluster_index (**int**): index of the corresponding cluster in the `clusters` dataset
 - event_id (**int**): event ID of edep-sim event (only for MC)

`ext_trig` (Optional): Note: External triggers from LRS trigger (average of the two external triggers per io group)
 - unix (**int**): unix timestamp of external trigger
 - t (**int**): PPS timestamp of external trigger

Datasets after charge-light matching stage:

`light_events`: Note: Light event timestamps and waveforms.
 - id (**int**): ID for light event
 - tai_ns (**int**): timestamp for light event in nsec
 - unix (**int**): unix second timestamp for light event
 - voltage_adc1 (**int**): waveforms for light event in ADC1
 - voltage_adc2 (**int**): waveforms for light event in ADC2

`header`:
 - channels_adc1 (**u1**): channel numbers for light event in ADC1
 - channels_adc2 (**u1**): channel numbers for light event in ADC2
 - max_hits (**int**): maximum hits allowed per cluster in an event
 - max_clusters (**int**): maximum number of clusters allowed per event
 - rate_threshold (**float**): threshold in Hz for disabling a channel
 - hit_threshold_LCM (**int**): threshold in ADCs for a summed waveform to be considered a 'hit' (LCMs)
 - hit_threshold_ACL (**int**): threshold in ADCs for a summed waveform to be considered a 'hit' (ACLs)

### Reconstruction inputs
There are multiple folders containing detector-specific configuration files:

- `config` folder: contains `json` files that have channel-by-channel v_ref and v_cm settings for the detector
- `detector_properties` folder: contains `yaml` files that have charge and light geometry parameters
- `disabled_channels` folder: contains `npz` files of channels to exclude in the reconstruction (e.g. noisy channels). Contains 1D arrays of keys (pixel unique id) and values
- `charge_layout` folder: detector-specific pixel layout yamls
- `light_layout` folder: light detector layout yamls
- `pedestal` folder: contains `json` files with channel-by-channel pixel pedestal values

### Charge to Light Matching
To match charge clusters to light, use the `match_clusters_to_light.py` script. Example:

```python
python3 match_light_to_clusters.py INFILENAME_CLUSTERS OUTFILENAME INFILENAME_LIGHT_1 INFILENAME_LIGHT_2 --input_config_name DET
```

It takes as input a clustered file made in the previous step, your preferred output filename, a file path for each .data LRS ADC file, and the input configuration name (e.g. module0).

### Analysis examples
To access the datasets in python:
```python
import h5py
import numpy as np

f = h5py.File('/path/to/file', 'r')
clusters = np.array(f['clusters'])
hits = np.array(f['hits'])
```

`plotting.py` includes many useful functions for plotting and analysis.
