#!/bin/bash

# Run 5 instances of the code for each set of groups for generation
# Two arguments in input, the group and the shift to apply alphas by

# We have 11259 alphas in total
# Shift counter on input by number of molecule combinations
# times 12 for each molecule combination
python GenerateTrainingData.py 0 0 &
python GenerateTrainingData.py 1 1440 &
python GenerateTrainingData.py 2 3720 &
# to not overrun no. of alphas, shift by negative difference between 12252 and 11259, so 8940-(12252-11259)
python GenerateTrainingData.py 3 7946 &
python GenerateTrainingData.py 4 0 &

