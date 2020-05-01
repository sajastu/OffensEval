Offensive Language Detection
==========

This repo contains the PyTorch code for Deep Learning project, Spring 2020, Georgetown University.


# Requirements

- Python 3.6
- PyTorch 1.1.0
- [Transformers](https://github.com/huggingface/transformers)
- [tqdm](https://github.com/tqdm/tqdm)

## Data
Download the clean data (OLID) under `dataset/` directory. To obtain further information about the dataset, please refer to [this paper](https://www.aclweb.org/anthology/N19-1144.pdf).

## Training

### Run training

To start training on  data, run:
```
python run.py --mode train --model_path $MODEL --data_dir $DS_BASE_DIR --task $TASK
```

where `$MODEL` is the base directory of the model that you want to train and the checkpoints will be saved in, `$DS_BASE_DIR` is the base directory of the dataset (default is `dataset/`), and `$TASK` is the sub-task index (choose from `a`, `b`, or `c`)
For other training parameters, you may refer to `other/utils.py`.

*Note:* You can start training by `sh train.sh` alternatively.

## Evaluation

*In progress...*