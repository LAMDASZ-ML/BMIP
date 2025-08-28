#!/bin/bash

#cd ../..

# custom config
DATA="/home/lvsl/DATA/Data"
TRAINER=BITP

DATASET=$1
SEED=$2
DEVICE=$3
RATIO=$4


TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

CFG=vit_b16_ep100  # config file
SUB=new
LOADEP=100

#    --output-dir output/base2new/train_${SUB}/${DATASET}/${CFG}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
#    --model-dir output/base2new/test_${SUB}/${DATASET}/${CFG}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \

COMMON_DIR=${DATASET}/${CFG}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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