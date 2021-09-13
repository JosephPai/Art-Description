#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 \
python caption_all.py \
    --model checkpoints/BEST_checkpoint_train_exp_01.pth.tar \
    --word_map dataset_semart/WORDMAP_SemArt_5_min_word_freq_120_max_len.json \
    --beam_size 5 \
    --output_file data/output/inference_testset.json

