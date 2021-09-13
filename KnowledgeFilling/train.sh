#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 \
python3 train_bert.py \
    --data_dir drqa_knowledge_filling_dataset \
    --bert_model bert-base-uncased \
    --learning_rate 8e-5 \
    --num_train_epochs 20 \
    --patience 4 \
    --device cuda \
    --batch_size 40 \
    --eval_batch_size 64 \
    --max_seq_length 512 \
    --workers 4 \
    --train_name BERT-KnowledgeFilling

