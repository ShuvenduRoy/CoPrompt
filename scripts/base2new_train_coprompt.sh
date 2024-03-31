#!/bin/bash

#cd ../..

# custom config
DATA=data/
TRAINER=CoPrompt

DATASET=$1
SEED=$2
EXP_NAME=$3 # base2new

CFG=coprompt
SHOTS=16

DIR=output/${EXP_NAME}/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
	--root ${DATA} \
	--seed ${SEED} \
	--trainer ${TRAINER} \
	--dataset-config-file configs/datasets/${DATASET}.yaml \
	--config-file configs/trainers/${CFG}.yaml \
	--output-dir ${DIR} \
	DATASET.NUM_SHOTS ${SHOTS} \
	DATASET.SUBSAMPLE_CLASSES base
