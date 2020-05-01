#!/usr/bin/env bash

MODEL=/disk1/sajad/trained-offns-bert/offns-a-sent/
DS_BASE_DIR=dataset/
TASK=a

rm -r $MODEL
mkdir -p $MODEL

python run.py --mode train --model_path $MODEL --data_dir $DS_BASE_DIR --task $TASK --device_id 1