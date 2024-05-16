#!/bin/bash
use_wandb=false
wandb_project=CS6910-Assignment3
wandb_entity=arjungangwar
in_embed_dims=128           # 16, 32, 64, 128
n_encoder_layers=1          # 1, 2, 3
n_decoder_layers=1          # 1, 2, 3
hidden_layer_size=128       # 16, 32, 64, 128, 256
cell_type=gru               # rnn, lstm, gru
bidirectional=0             # 1: yes, 0: no
dropout=0.2
n_epochs=15
learning_rate=1e-4
max_length=25
#beam_search=5

opts=
if ${use_wandb}; then 
    opts+="--use_wandb "
fi

python train.py \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --in_embed_dims ${in_embed_dims} \
    --n_encoder_layers ${n_encoder_layers} \
    --n_decoder_layers ${n_decoder_layers} \
    --hidden_layer_size ${hidden_layer_size} \
    --cell_type ${cell_type} \
    --bidirectional ${bidirectional} \
    --dropout ${dropout} \
    --n_epochs ${n_epochs} \
    --learning_rate ${learning_rate} \
    --max_length ${max_length} \
    ${opts}