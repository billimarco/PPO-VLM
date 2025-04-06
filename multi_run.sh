#!/bin/bash

# Parametri fissi
EXP_NAME_BASE="ppo_swin_hf"
NETWORK_TYPE="swin_w_hf"
PRETRAINED_ADAPT="True"
TRACK="True"
LRL="lrl--_adapted"

# Percorso al main.py (modifica se necessario)
SCRIPT_PATH="main.py"

# Loop sui parametri
for seed in 1 2 3; do
  for forward_type in "single_frame" "multi_frame_patch_concat"; do
    for use_lora in True False; do

      # Salta la combinazione proibita
      if [ "$seed" -eq 1 ] && [ "$forward_type" = "single_frame" ] && [ "$use_lora" = "False" ]; then
        echo "⚠️  Skipping forbidden combination: seed=1, forward_type=single_frame, use_lora=False"
        continue
      fi

      # Aggiungi _lora se serve
      if [ "$use_lora" = "True" ]; then
        LORA_TAG="_lora"
      else
        LORA_TAG=""
      fi

      # Costruzione del nome esperimento
      EXP_NAME="${EXP_NAME_BASE}_${seed}_${forward_type}_${LRL}${LORA_TAG}"

      echo "== Running seed=${seed}, forward=${forward_type}, lora=${use_lora} =="
      python ${SCRIPT_PATH} \
        --track ${TRACK} \
        --seed ${seed} \
        --exp-name "${EXP_NAME}" \
        --network-type "${NETWORK_TYPE}" \
        --pretrained-adapt ${PRETRAINED_ADAPT} \
        --forward-type "${forward_type}" \
        --use-lora ${use_lora}
      
      echo ""
    done
  done
done