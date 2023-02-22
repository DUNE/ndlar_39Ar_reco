*this repository is under-construction*

This repository contains code for reconstructing point-like energy deposits in ND-LAr prototypes data and simulations. The code will also automatically find large clusters (e.g. cosmic tracks). At the moment, the code only supports Module-0. There is a fork of larnd-sim (ND-LAr simulation software) that allows for using the LAr NEST model for recombination, which may be more relevant for very low energy electrons, This repository and code are still a work in-progress.

Currently, the code starts out by using DBSCAN to cluster hits to find tracks. Then the tracks are "thrown away" and a second-round of clustering is performed on the hits that remain (non-track-like clusters) to find small clusters of hits. What results is many clusters of a few hits (depending on the clustering). The ADCs of the hits are converted to charge and summed within clusters. The results are saved to an h5py file for both small and large clusters. 

Things I'm working on implementing:
1. Output the results of the reconstruction to a light-weight h5py file
2. Charge-light matching between the reconstructed charge events and light data
3. Expand support for other single ND-LAr modules and the 2x2 Demonstrator

To clone LArNDLE, run:
```bash
git clone https://github.com/sam-fogarty/LArNDLE.git
cd LArNDLE
pip install .
```

Here is an example of running the reconstruction in command-line:

```python
python3 reco/reco.py \
--input_packets_filename=datalog_2021_04_04_00_41_40_CEST.h5 \
--output_events_filename=datalog_2021_04_04_00_41_40_CEST_events.h5 \
--nSec_start=30 \
--nSec_end = 60
```

nSec is the number of seconds of data to process. Each second of data (between PPS pulses) is processed individually and the results (events) are all concatenated together. The output is an h5 files containing two datasets, `small_clusters` and `large_clusters`. The former will contain point-like events (few hits) while the latter contains larger cluster events (e.g. cosmic tracks). The data format will surely change slightly as the code continues to be developed.

LArNDLE needs a dictionary that can retrieve the pixel positions corresponding to larpix hits. This dictionary is taken in the form of a pickle file, which is made with `larpix_readout_parser` (https://github.com/YifanC/larpix_readout_parser). Look in the `layout` folder, because there might already be the one you need there (for example, multi_tile_layout-2.3.16.pkl is for module-0).

You can find my larnd-sim fork here, if you want to use the NEST model for recombination:
https://github.com/sam-fogarty/larnd-sim_beta-decays
Note that larnd-sim is not a dependency of LArNDLE. But you may install larnd-sim to produce simulation samples to run through LArNDLE. In that case, you don't have to use my fork unless you want the option to use the NEST model.
