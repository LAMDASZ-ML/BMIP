#!/bin/bash

#cd ../..

# custom config
DATA="/mnt/hdd/DATA"
TRAINER=CoOp

DATASET=$1
CFG=vit_b16_ep100  # config file
SHOTS=16
NCTX=16
CSC=False
CTP=end
SUB=base

for SEED in 1 2 3
do
    DIR=/mnt/hdd/lvsl/output/base2new/train_${SUB}/${DATASET}/${CFG}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done