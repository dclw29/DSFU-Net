#!/bin/bash

START=0
END=1008

for ((i=${START};i<=${END};i+=36)); do
    #sbatch /scratch/dclw/ML-DiffuseReader/main/gen_data.slurm $i $(($i+36))
    python /data/lrudden/ML-DiffuseReader/main/Generate_TrainingData.py $i $(($i+36)) > logfiles/tmp 2>logfiles/tmp &
done

python /data/lrudden/ML-DiffuseReader/main/Generate_TrainingData.py 1044 1081 > logfiles/tmp 2>logfiles/tmp &

#sbatch /scratch/dclw/ML-DiffuseReader/main/gen_data.slurm 1044 1081

# the very last loop needs hand selecting because of the remainder (i.e. 1081 molecules can't be divided into 36 chunks evenly)
