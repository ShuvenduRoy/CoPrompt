#!/bin/bash

#cd ../..

# custom config
DATA={YOUR_DATA_PATH}
TRAINER=CoPrompt

DATASET=$1
SEED=$2
EXP_NAME=$3

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=$4
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_randaug/seed${SEED}
MODEL_DIR=output/${EXP_NAME}/train_base/${COMMON_DIR}
DIR=output/${EXP_NAME}/test_${SUB}/${COMMON_DIR}

echo "Runing the first phase job and save the output to ${DIR}"

python train.py \
	--root ${DATA} \
	--seed ${SEED} \
	--trainer ${TRAINER} \
	--dataset-config-file configs/datasets/${DATASET}.yaml \
	--config-file configs/trainers/${CFG}.yaml \
	--output-dir ${DIR} \
	--model-dir ${MODEL_DIR} \
	--load-epoch ${LOADEP} \
	--eval-only \
	DATASET.NUM_SHOTS ${SHOTS} \
	DATASET.SUBSAMPLE_CLASSES ${SUB}
