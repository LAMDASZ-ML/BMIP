#!/bin/bash

#cd ../..

# custom config
DATA="/data0/lvsl/DATA"
TRAINER=BITP

DATASET=$1
SEED=$2
DEVICE=$3
CFG=$4
load_epoch=$5

SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --device ${DEVICE} \
    --load-epoch ${load_epoch} \
    --eval-only
fi