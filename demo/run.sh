#!/bin/bash

input=INPUT/test0.npy
EPOCH=200 # default anyway
output=OUTPUT

python ../main/pipeline.py --filename ${input} --generator_loc ../models/ --epoch ${EPOCH} --no_norm

mv ./INPUT/*SRO* ./INPUT/*dFF* ${output}
