This repository contains code for reconstructing energy depositsin ND-LAr prototypes data and simulations. It uses DBSCAN to find clusters of various sizes and saves cluster-level info (number of hits, total charge, etc) and hit-level info (charge, position, timestamp, io group, etc) to an h5py output file. The clustering algorithm will find clusters of various sizes, anywhere from single hits to large tracks or showers. This includes energy deposits from radiologicals (e.g. 39Ar), cosmic tracks, beam spills, etc. One can run an analysis on the output file of this code to further isolate events of interest. 

Charge-light matching between LRS triggers and clusters is supported for data (can be toggled on or off). This requires one `.data` file per ADC in the detector (This may change in future versions). The data associated with matched LRS triggers is saved to the output file. Associations between clusters, hits, and LRS triggers are maintained in the datasets. The light dataset includes unix and PPS timestamps, and all waveforms and channels associated with each trigger. The corresponding clusters dataset contains a `matched` parameter which is 0 if the charge cluster was not matched to a light trigger and 1 if it was. The `light_index` parameter contains the index of the matched light trigger within the corresponding light triggers dataset. Note that light triggers not matched to a cluster are not saved in this file, but all clusters are saved to the output file regardless of whether they were matched to light.

To clone the code, run:
```bash
git clone https://github.com/sam-fogarty/ndlar_hitfinder.git
cd ndlar_hitfinder
pip install .
```

Here is an example of running the reconstruction in command-line, once cd'd into the `reco` directory:

```python
python3 reco.py detector.py
```

`detector.py` is an input config file stored in the `input_config` directory. These files contain input variables for `reco.py`. This allows for easily running different configurations, especially when running a different detector (e.g. module-0,1,2,3 and 2x2) or switching between data and simulation. This code should mostly be agnostic to the different detectors at this point, but may need slight adjustments to work properly for new detectors.

The numbers of seconds of data to process is set in the input config file. Each second of data (between Pulse Per Second (PPS) pulses) is processed individually and the results are concatenated together. When processing data, there are various progress bars to give you an idea of how long it will take to process the file. Note that for simulation no such progress bars are currently supported. Note that currently the code assumes that the timestamps are reset every second via a PPS pulse, and will not work without it. This will be rectified in a future version. 

Note that the matching tolerance values may need to be adjusted in order to ensure that the light triggers get matched to the external triggers (packet type 7's within the packets data). Once this matching has been done, we match light events to charge events based on the unix and PPS timestamps. In particular, by default they match if they're within 1s in unix and one drift window in PPS.

The code needs a dictionary that can retrieve the pixel positions corresponding to larpix hits. This dictionary is taken in the form of a pickle file, which is made with `larpix_readout_parser` (https://github.com/YifanC/larpix_readout_parser). Look in the `layout` folder, because there might already be the one you need there (for example, multi_tile_layout-2.3.16.pkl is for module-0).
