#!/bin/bash

#cd ../..

# custom config
DATA="/home/lvsl/DATA/Data"
TRAINER=MaPLe

DATASET=$1
SEED=$2
DEVICE=$3
RATIO=$4


CFG=vit_b16_c2_ep50_ctx3_dp9TF
SHOTS=16
LOADEP=15
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}/RATIO_${RATIO}

echo "Evaluating model"
echo "Calculating the different ratio of data"
echo "Runing the first phase job and save the output to ${DIR}"


echo "Current ratio is ${RATIO}"
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
--best \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB} \
DATASET.SUBSAMPLE_RATIO ${RATIO}

