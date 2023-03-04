*this repository is under-construction*

This repository contains code for reconstructing point-like energy deposits in ND-LAr prototypes data and simulations. The code will also automatically find large clusters (e.g. cosmic tracks). At the moment, the code only supports Module-0 and Module-3.

Currently, the code starts out by using DBSCAN to cluster hits to find tracks. Thena second-round of clustering is performed on the hits that remain (non-track-like clusters, or hits that were not clustered in the first step) to find small clusters of hits. What results is many clusters of a few hits (depending on the clustering parameters). The ADCs of the hits are converted to charge and summed within clusters. The results are saved to an h5py file for both small and large clusters. Charge-light matching is implemented for small clusters and large clusters (can be toggled on or off in `consts.py`).

To clone LArNDLE, run:
```bash
git clone https://github.com/sam-fogarty/LArNDLE.git
cd LArNDLE
pip install .
```

Here is an example of running the reconstruction in command-line, once cd'd into the `reco` directory:

```python
python3 reco.py moduleX.py
```

`moduleX.py` is an input config file stored in the `input_config` directory. These files contain input variables for `reco.py`. This allows for easily running different configurations, especially when running a different detector (i.e., module-0,1,2,3 and 2x2). This code should mostly be agnostic to the different detectors at this point, but may need slight adjustments to work properly.

The numbers of seconds of data to process is set in the input config file. Each second of data (between PPS pulses) is processed individually and the results (events) are concatenated together. The output is an h5 file containing four datasets, `small_clusters`, `large_clusters`, `small_clusters_matched_light`, and `large_clusters_matched_light`. `small_clusters` will contain point-like events (few hits) while `large_clusters` contains larger cluster events (e.g. cosmic tracks, large noise events). The matched light datasets contain light events, which includes unix and PPS timestamps, and all waveforms and channels. The corresponding charge datasets contain a `matched` parameter which is 0 if the charge event was not matched to a light event and 1 if it was. The `light_index` parameter contains the index of the matched light event within the corresponding light events dataset. So one can easily get the charge events associated with a specific light trigger. The data format will surely change slightly as the code continues to be developed. Note that light events not matched to a charge event are not saved in this file.

Note that the matching tolerance values may need to be adjusted in order to ensure that the light triggers get matched to the external triggers (packet type 7's within the packets data). Once this matching has been done, we match light events to charge events based on the unix and PPS timestamps. In particular, by default they match if they're within 1s in unix and one drift window in PPS.

There are a handful of data cuts available in `charge_event_cuts`.

LArNDLE needs a dictionary that can retrieve the pixel positions corresponding to larpix hits. This dictionary is taken in the form of a pickle file, which is made with `larpix_readout_parser` (https://github.com/YifanC/larpix_readout_parser). Look in the `layout` folder, because there might already be the one you need there (for example, multi_tile_layout-2.3.16.pkl is for module-0).