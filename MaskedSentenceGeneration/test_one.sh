#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 \
python caption_one.py \
    --img ../KnowledgeRetrieval/context_art_classification/Data/SemArt/Images/43921-penn.jpg \
    --model checkpoints/BEST_checkpoint_train_exp_01.pth.tar \
    --word_map dataset_semart/WORDMAP_SemArt_5_min_word_freq_120_max_len.json \
    --beam_size 5

