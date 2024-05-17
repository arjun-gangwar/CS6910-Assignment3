#!/bin/bash
use_wandb=true
wandb_project=CS6910-Assignment3
wandb_entity=arjungangwar
in_embed_dims=128           # 32, 64, 128
n_layers=3                  # 1, 2, 3
hidden_layer_size=128       # 32, 64, 128, 256
cell_type=lstm               # rnn, lstm, gru
bidirectional=true         # true, false
dropout=0.2
n_epochs=15
learning_rate=1e-3
max_length=25
batch_size=32

opts=
if ${use_wandb}; then 
    opts+="--use_wandb "
fi
if ${bidirectional}; then 
    opts+="--bidirectional "
fi

python train.py \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --in_embed_dims ${in_embed_dims} \
    --n_layers ${n_layers} \
    --hidden_layer_size ${hidden_layer_size} \
    --cell_type ${cell_type} \
    --dropout ${dropout} \
    --n_epochs ${n_epochs} \
    --learning_rate ${learning_rate} \
    --max_length ${max_length} \
    --batch_size ${batch_size} \
    ${opts}