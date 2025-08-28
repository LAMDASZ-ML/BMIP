#!/bin/bash

#cd ../..

# custom config
DATA="/mnt/hdd/DATA"

DATASET=$1
SEED=$2
DEVICE=$3
INIT_SEED=$4
TRAINER=PromptSRC_BMIP

CFG=vit_bitp_ctx4_ep20TT_adam_PromptSRC
SHOTS=16
LOADEP=20
SUB=new


COMMON_DIR=${DATASET}/${CFG}_${INIT_SEED}/seed${SEED}
MODEL_DIR=output/base2new/train_base_SEED/${COMMON_DIR}
DIR=output/base2new/test_${SUB}_SEED/${COMMON_DIR}
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
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.INIT_SEED ${INIT_SEED}

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
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.INIT_SEED ${INIT_SEED}
fi