#!/bin/bash

#cd ../..

# custom config
DATA="/mnt/hdd/DATA"
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=$1
CFG=vit_b16_ep100  # config file
SUB=new

BASE_DIR=/mnt/hdd/lvsl/output/base2new/
for SEED in 1 2 3
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${BASE_DIR}/test_new/${DATASET}/${CFG}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --model-dir ${BASE_DIR}/train_base/${DATASET}/${CFG}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 100 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.SUBSAMPLE_CLASSES new
done