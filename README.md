# strath_UCAM_MET4FOF

UCAM analysis of STRATH dataset for MET4FOF

## Data source 
Sensor data set radial forging at AFRC testbed
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2573861.svg)](https://doi.org/10.5281/zenodo.2573861)

**requires pytorch for ANN models

## Code information:
1. loadFullSensorsData.py - load,preprocess,plots,pickling
2. ANN.py - modeling, kfold, 
3. BNN.py - same as above, but with uncertainty plots & quantification using dropout as Bayesian approx.
4. time_segmentation.py - WIP
