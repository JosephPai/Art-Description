#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --exp_name train_exp_01 \
    --data_folder dataset_semart \
    --batch_size 32 \
    --emb_dim 512 \
    --attention_dim 512 \
    --decoder_dim 512 \
    --encoder_lr 1e-4 \
    --decoder_lr 1e-3 \
    --fine_tune_encoder True \

