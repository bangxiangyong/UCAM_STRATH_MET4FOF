# UCAM AFRC MET4FOF

UCAM analysis of STRATH-AFRC dataset for MET4FOF

## Data source 
Sensor data set radial forging at AFRC testbed
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2573861.svg)](https://doi.org/10.5281/zenodo.2573861)

## Code information:
1. 01-Load-Data.ipynb - Download & parse data into dataframes
2. 02-Time-Segmentation.ipynb - Segment the sensor measurements into 3 phases : Heating, Transfer, Forging (and an additional Full which comprises of all 3 phases)
3. 03-Feature-Extraction.ipynb - Extract features from time-series: Mean, Std, Kurtosis, Min, Max, Skewness,Sum,Median
4. 04-BNN-Dropout.ipynb - Bayesian Neural Network using Dropout for modelling. Kfold validation and uncertainty quantification.

## Note: 
1. Thanks to Christos Tachtatzis & Yuhui Luo 
2. Requires pytorch for ANN models
