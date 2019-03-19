# strath_UCAM_MET4FOF

UCAM analysis of STRATH dataset for MET4FOF

##Placement of dataset
1. scope001-0080.csv dataset folder: ../EMPIR_Data/Data/AFRC Radial Forge - Historical Data
2. CMMdataset.xlsx folder: ../EMPIR_Data/Data
3. ScopeDataWithHeadings.xlsx folder: ../EMPIR_Data

**requires pytorch for ANN models

##Code information:
1. loadFullSensorsData.py - load,preprocess,plots,pickling
2. ANN.py - modeling, kfold, 
3. BNN.py - same as above, but with uncertainty plots & quantification using dropout as Bayesian approx.
4. time_segmentation.py - WIP
