#!/bin/bash

#cd ../..

# custom config
DATA="/mnt/hdd/DATA"
TRAINER=MaPLe

DATASET=$1
SEED=$2
DEVICE=$3
BEST=$4

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5
SUB=new

# DIR=/mnt/hdd/lvsl/output/base2new/train_base/MaPLe_${CFG}/${DATASET}/seed${SEED}
# COMMON_DIR=${CFG}_BEST/${DATASET}/seed${SEED}
MODEL_DIR=/mnt/hdd/lvsl/output/base2new/train_base/MaPLe_${CFG}/${DATASET}/seed${SEED}
DIR=/mnt/hdd/lvsl/output/base2new/test_${SUB}/MaPLe_${CFG}/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --device ${DEVICE} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --device ${DEVICE} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi