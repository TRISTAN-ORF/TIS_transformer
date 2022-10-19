#!/bin/bash

transcript_transformer train 'data.json' --val 2 14 --test 1 7 13 19 --gpus 1 --max_epochs 60 --name 'TIS_trans_1' --dim 48 --depth 8 --heads 8 --dim_head 16 --local_attn_heads 5
transcript_transformer train 'data.json' --val 1 13 --test 2 8 14 20 --gpus 1 --max_epochs 60 --name 'TIS_trans_2' --dim 48 --depth 8 --heads 8 --dim_head 16 --local_attn_heads 5
transcript_transformer train 'data.json' --val 1 13 --test 3 9 15 21 --gpus 1 --max_epochs 60 --name 'TIS_trans_3' --dim 48 --depth 8 --heads 8 --dim_head 16 --local_attn_heads 5
transcript_transformer train 'data.json' --val 1 13 --test 4 10 16 22 --gpus 1 --max_epochs 60 --name 'TIS_trans_4' --dim 48 --depth 8 --heads 8 --dim_head 16 --local_attn_heads 5
transcript_transformer train 'data.json' --val 1 13 --test 5 11 17 X --gpus 1 --max_epochs 60 --name 'TIS_trans_5' --dim 48 --depth 8 --heads 8 --dim_head 16 --local_attn_heads 5
transcript_transformer train 'data.json' --val 1 13 --test 6 12 18 Y --gpus 1 --max_epochs 60 --name 'TIS_trans_6' --dim 48 --depth 8 --heads 8 --dim_head 16 --local_attn_heads 5