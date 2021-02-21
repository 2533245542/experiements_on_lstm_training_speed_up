## Introduction
We examine two factors: batch size and GPU/CPU.

## Procedures
Modify the TODO's in `examine_factors_that_impact_lstm_training_speed.py` to run the experiments.

`python examine_factors_that_impact_lstm_training_speed.py`

## Results
Batch size has a significant impact: batch size changes from 1 to 160 -> speed up for 40 fold.

GPU/CPU does not have a significant impact (to switch from CPU to GPU) and it might be due to the low parallelization of recurrent neural network. 

Switching from CPU to GPU when batch size = 160 -> speed up of 1.6 fold. 

Te number of speed up folds decrease when batch size is lower (lower parallelization); when batch size = 1, the speed up folds decreases to 1.10.