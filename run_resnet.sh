#!/bin/bash

# Array dei learning rate
learning_rates=("2.5e-5")

# Array dei num minibatches
minibatches=(8)

# Parametri fissi
base_exp_name="ppo_resnet_1_single_frame_1500upd_adapted"
fixed_args="--track True --seed 1 --network-type resnet_w --ac-mlp False --pretrained-adapt True --forward-type single_frame --updates-per-env 1500"

# Itera su ogni combinazione
for lr in "${learning_rates[@]}"; do
  for mb in "${minibatches[@]}"; do
    # Costruisci un nome esperimento univoco
    exp_name="${base_exp_name}_lr${lr}_mb${mb}"

    # Esegui il comando
    python main.py $fixed_args --learning-rate $lr --num-minibatches $mb --exp-name "$exp_name"
  done
done