#!/bin/bash

# Array dei learning rate
learning_rates=("1.0e-3" "2.5e-4")

# Array dei num minibatches
minibatches=(4 8)

# Parametri fissi
base_exp_name="ppo_cnn_3_single_frame_adapted"
fixed_args="--track True --seed 3 --network-type cnn --ac-mlp False --pretrained-adapt True --forward-type single_frame"

# Itera su ogni combinazione
for lr in "${learning_rates[@]}"; do
  for mb in "${minibatches[@]}"; do
    # Costruisci un nome esperimento univoco
    exp_name="${base_exp_name}_lr${lr}_mb${mb}"

    # Esegui il comando
    python main.py $fixed_args --learning-rate $lr --num-minibatches $mb --exp-name "$exp_name"
  done
done