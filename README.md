# strath_UCAM_MET4FOF

UCAM analysis of STRATH dataset for MET4FOF

-placement of dataset: loading files require folders and right placing of files, due to large size, dataset is not uploaded here. 
*scope001-0080.csv dataset folder: ../EMPIR_Data/Data/AFRC Radial Forge - Historical Data
*CMMdataset.xlsx folder: ../EMPIR_Data/Data
*ScopeDataWithHeadings.xlsx folder: ../EMPIR_Data

-requires pytorch for ANN models

Code information:
1. loadFullSensorsData.py - load,preprocess,plots,pickling
2. ANN.py - modeling, kfold, 
3. BNN.py - same as above, but with uncertainty plots & quantification using dropout as Bayesian approx.
4. time_segmentation.py - WIP
