#/bin/bash

EXPT_NAME="RETRAIN"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python ${SCRIPT_DIR}/dsfu-net.py --gpu 0 --batch_size 64 --experiment_name ${EXPT_NAME} >> run_${EXPT_NAME}.log
