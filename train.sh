#!/usr/bin/env bash

MODEL=/disk1/sajad/trained-offns-bert/offns-mtl-pretrained/
DS_BASE_DIR=dataset/
TASK=all
#TASK=a
CH=/disk1/sajad/pretrained-bert/pretraining_output/
#CH=bert-base-uncased

rm -r $MODEL
mkdir -p $MODEL

python run.py --mode train --init_checkpoint $CH --model_path $MODEL --data_dir $DS_BASE_DIR --task $TASK --device_id 0