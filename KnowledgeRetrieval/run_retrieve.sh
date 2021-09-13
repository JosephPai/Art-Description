#!/usr/bin/env bash
python retrieve_knowledge.py \
  --dataset words_test_set.json \
  --out-file retrieved_paragraph_testset_semart_src.json \
  --gpu 0 \
  --n-docs 10 \
  --top-n 10 \
  --num-workers 8 \
