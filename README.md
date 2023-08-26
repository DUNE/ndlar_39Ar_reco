### Overview
This code is tailored to reconstructing energy deposits from radiologicals in the ND-LAr prototypes (data and simulation). It uses DBSCAN to find clusters of charge hits. Since the code finds clusters of various sizes, one can make a selection on the output file to find small clusters corresponding to radiological deposits. Charge clusters are matched to external trigger packets that are injected into the charge system when there is a light readout trigger. This provides a t0 for the clusters that is used to calculate the drift coordinate (z_drift). The way this works is for every external trigger packet, the code finds any clusters with timestamps within an asymmetric window around the external trigger PPS timestamp (which includes charge that occurs within a drift window).

### Setting up the code
To setup the code, run:
```bash
git clone https://github.com/sam-fogarty/ndlar_39Ar_reco.git
cd ndlar_39Ar_reco
pip install .
```

Here is an example of running the reconstruction on the commandline. First `cd` to the `charge_reco` directory, then:
```python
python3 charge_clustering.py module-0 /path/to/input/packet/h5/file /path/to/output/h5/file
```
`module-0` an option inside in the ModuleConfig class inside `input_config.py`. It contains detector-specific configuration parameters. Make sure to use the correct detector configuration for the chosen input file. The input is a packetized h5 file containing the charge data. The output must be an h5 file. The reconstruction is done in batches (configurable) and a progress bar is shown. Note that if you choose to save the `hits` dataset, the output file will be much larger. 

### Output Format
The output file contains the following datasets:

`clusters` : Note: Charge clusters found with DBSCAN 
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
 - ext_trig_index (**int**): index of matched ext. trig in external trigger dataset

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

### Reconstruction inputs
The code requires various inputs, most of which are specified in the input configuration files in the `input_config` folder. 

- `config` folder: contains `json` files that have channel-by-channel v_ref and v_cm settings for the detector
- `detector_properties` folder: contains `yaml` files that have charge and light geometry parameters
- `disabled_channels` folder: contains `npz` files of channels to exclude in the reconstruction (e.g. noisy channels). Contains 1D arrays of keys (pixel unique id) and values
- `layout` folder: detector-specific pixel layout yamls
- `pedestal` folder: contains `json` files with channel-by-channel pixel pedestal values

### Analysis examples
To access the datasets in python:
```python
import h5py
import numpy as np

f = h5py.File('/path/to/file', 'r')
clusters = np.array(f['clusters'])
hits = np.array(f['hits'])
ext_trig = np.array(f['ext_trig'])
```
To find small clusters (e.g. radiologicals):
```python
nhit_cut = 10
cluster_nhit = np.array(clusters['nhit'])
small_clusters = clusters[cluster_nhit <= nhit_cut]
```
You can find more analysis examples and information pertaining to radiological simulation [here](https://github.com/sam-fogarty/ndlar_39Ar_analysis).
