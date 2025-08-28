#!/bin/bash

#cd ../..

# custom config
DATA="/home/lvsl/DATA/Data"
TRAINER=VPT

DATASET=$1
SEED=$2
DEVICE=$3
RATIO=$4


CFG=vit_b16_c2_ep5_batch4_4
SHOTS=16
LOADEP=5
SUB=new

#/home/lvsl/Code/BAITP/output/base2new/train_base/caltech101/vit_b16_c2_ep5_batch4_4/VPT/seed1/prompt_learner
COMMON_DIR=${DATASET}/${CFG}/${TRAINER}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/RATIO/${COMMON_DIR}/RATIO_${RATIO}
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
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB} \
DATASET.SUBSAMPLE_RATIO ${RATIO}
#echo "Finish the first phase job"
#kill -9 $(ps -ef | grep "base2new_ratio1" | grep -v grep | awk '{print $2}')