# strath_MET4FOF

UCAM analysis of STRATH dataset for MET4FOF

-requires pytorch

1. loadFullSensorsData.py - load,preprocess,plots,pickling
2. ANN.py - modeling, kfold, 
3. BNN.py - same as above, but with uncertainty plots & quantification using dropout as Bayesian approx.
4. time_segmentation.py - WIP
