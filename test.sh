#!/usr/bin/env bash

TASK=B
MODEL=/disk1/sajad/trained-offns-bert/offns-b-sent/
#CHECKPOINT=mtl_BEST_Task_$TASK.pt
CHECKPOINT=BEST_Task_b.pt
DS_BASE_DIR=dataset/
INIT_CH=/disk1/sajad/pretrained-bert/pretraining_output/
#INIT_CH=bert-base-uncased

python run.py --mode test --init_checkpoint $INIT_CH --saved_model $MODEL/$CHECKPOINT  --data_dir $DS_BASE_DIR --task $TASK --device_id 0