#!/bin/bash

window=3
hidden_size=1024
emb_size=5
lr=1e-3
batch_size=64
epochs=10
seed=42

vocab_path="model_data/data/vocabulary.pkl"
data_path="model_data/data/processed_data_w$window.pkl"
save_model_path="model_data/NPLM-w$window-emb$emb_size-h$hidden_size-bs$batch_size-lr$lr"
load_model_path="model_data/NPLM-w$window-emb$emb_size-h$hidden_size-bs$batch_size-lr$lr-10.pth"

eval_test="--eval-test"
eval_test=""

python -u Embeddings.py \
  $vocab_path \
  $save_model_path \
  $window \
  $hidden_size \
  $emb_size \
  --data-path=$data_path \
  --load-model-path=$load_model_path \
  --seed=$seed \
  --batch-size=$batch_size \
  --lr=$lr \
  --epochs=$epochs \
  $eval_test
