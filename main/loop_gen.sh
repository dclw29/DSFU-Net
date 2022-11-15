#!/bin/bash

START=0
END=371

CPU_START=63
CPU_END=$(($CPU_START+7))

for ((i=${START};i<=${END};i+=53)); do
    #sbatch /scratch/dclw/ML-DiffuseReader/main/gen_data.slurm $i $(($i+36))
    taskset --cpu-list ${CPU_START}-${CPU_END} python /data/lrudden/ML-DiffuseReader/main/Generate_TrainingData.py $i $(($i+53)) > logfiles/tmp 2>logfiles/tmp &
    CPU_START=$(($CPU_END+1))
    CPU_END=$(($CPU_START+7))
done

python /data/lrudden/ML-DiffuseReader/main/Generate_TrainingData.py 371 430 > logfiles/tmp 2>logfiles/tmp &

#sbatch /scratch/dclw/ML-DiffuseReader/main/gen_data.slurm 1044 1081

# the very last loop needs hand selecting because of the remainder (i.e. 1081 molecules can't be divided into 36 chunks evenly)
