#!/usr/bin/env bash

TASK=a
MODEL=/disk1/sajad/trained-offns-bert/offns-a-sent/
CHECKPOINT=BEST_Task_$TASK.pt
DS_BASE_DIR=dataset/

python run.py --mode test --saved_model $MODEL/$CHECKPOINT  --data_dir $DS_BASE_DIR --task $TASK --device_id 1