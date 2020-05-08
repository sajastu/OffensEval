#!/usr/bin/env bash

MODEL=/disk1/sajad/trained-offns-bert/offns-c/
DS_BASE_DIR=dataset/
TASK=c

rm -r $MODEL
mkdir -p $MODEL

python run.py --mode train --init_checkpoint bert-base-uncased --model_path $MODEL --data_dir $DS_BASE_DIR --task $TASK --device_id 1