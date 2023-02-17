*this repository is under-construction*

This repository contains code for reconstructing point-like energy deposits in ND-LAr prototypes data and simulations. At the moment, the code only supports Module-0. There is a fork of larnd-sim (ND-LAr simulation software) that allows for using the LAr NEST model for recombination, which may be more relevant for very low energy events like 39Ar beta decays. The repository (and code) is still a work in-progress.

Currently, the code starts out by using DBSCAN to cluster hits to find tracks. Then the tracks are "thrown away" and a second-round of clustering is performed on the hits that remain (non-track-like clusters) to find small clusters of hits. What results is many clusters of a few hits (depending on the clustering). The ADCs of the hits are converted to charge and summed within clusters. 

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

To run the reconstruction in command-line, you can run:

```python
python3 reco/reco.py \
--input_packets_filename=datalog_2021_04_04_00_41_40_CEST.h5 \
--selection_start=0 \
--selection_end=10000 \
```

A selection of the packets data is made to run the reconstruction on a small range of data. This is a temporary method, I'll update with a better way.

You can find my larnd-sim fork here:
https://github.com/sam-fogarty/larnd-sim_beta-decays
Note that larnd-sim is not a dependency of LArNDLE. But you may install larnd-sim to produce simulation samples to run through LArNDLE. In that case, you don't have to use my fork unless you want the option to use the NEST model.
