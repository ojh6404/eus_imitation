#!/usr/bin/env bash

DEFAULT_DATASET_PATH="$HOME/tensorflow_datasets"
IMITATOR_PATH="$HOME/.imitator"
HF_CACHE_PATH="$HOME/.cache/huggingface"

if [ -z "$1" ]; then
  DATASET_PATH=$DEFAULT_DATASET_PATH
else
  DATASET_PATH=$1
fi

# run docker
docker run -it --rm \
  --gpus all \
  --name imitator \
  -v $IMITATOR_PATH:/home/user/.imitator \
  -v $DATASET_PATH:/home/user/tensorflow_datasets \
  -v $HF_CACHE_PATH:/home/user/.cache/huggingface \
  --net=host \
  eus_imitation:latest \
  /bin/bash
