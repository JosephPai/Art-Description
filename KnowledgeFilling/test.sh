#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 \
python3 test_bert.py \
    --bert_model bert-base-uncased \
    --device cuda \
    --eval_batch_size 64 \
    --max_seq_length 512 \
    --workers 0 \
    --train_name BERT-KnowledgeFilling

