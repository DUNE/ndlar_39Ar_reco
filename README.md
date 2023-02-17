*this repository is under-construction*

This repository contains code for reconstructing point-like energy deposits in ND-LAr prototypes data and simulations. At the moment, the code only supports Module-0. There is a fork of larnd-sim (ND-LAr simulation software) that allows for using the LAr NEST model for recombination, which may be more relevant for very low energy events like 39Ar beta decays. The code is still a work in-progress.

Currently, the code starts out by using DBSCAN to cluster hits to find tracks. Then the tracks are "thrown away" and a second-round of clustering is performed on the hits that remain (non-track-like clusters) to find small clusters of hits. What results is many clusters of a few hits (depending on the clustering). The ADCs of the hits are converted to charge and summed within clusters. 

Things I'm working on implementing:
1. Output the results of the reconstruction to a light-weight h5py file
2. Charge-light matching between the reconstructed charge events and light data
Plan on doing:
3. Expand support for other single ND-LAr modules and the 2x2 Demonstrator

You can find my larnd-sim fork here:
https://github.com/sam-fogarty/larnd-sim_beta-decays


