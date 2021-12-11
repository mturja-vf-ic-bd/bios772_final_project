#!/bin/bash

python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_100_17 --template_list vol_schaefer_100_17 thick_schaefer_100_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_200_17 --template_list vol_schaefer_200_17 thick_schaefer_200_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_300_17 --template_list vol_schaefer_300_17 thick_schaefer_300_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_400_17 --template_list vol_schaefer_400_17 thick_schaefer_400_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_500_17 --template_list vol_schaefer_500_17 thick_schaefer_500_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_600_17 --template_list vol_schaefer_600_17 thick_schaefer_600_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_700_17 --template_list vol_schaefer_700_17 thick_schaefer_700_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_800_17 --template_list vol_schaefer_800_17 thick_schaefer_800_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_900_17 --template_list vol_schaefer_900_17 thick_schaefer_900_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_schaefer_1000_17 --template_list vol_schaefer_1000_17 thick_schaefer_1000_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_aal --template_list vol_aal thick_aal --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_aal2 --template_list vol_aal2 thick_aal2 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_aal3 --template_list vol_aal3 thick_aal3 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_yeo7 --template_list vol_yeo_7 thick_yeo_7 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_yeo17 --template_list vol_yeo_17 thick_yeo_17 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_brodmann --template_list vol_Brodmann thick_Brodmann --max_epochs 5000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1 &
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_gordon --template_list vol_Gordon thick_Gordon --max_epochs 5000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_hammers_83 --template_list vol_Hammers_83 thick_Hammers_83 --max_epochs 5000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_hammers_95 --template_list vol_Hammers_95 thick_Hammers_95 --max_epochs 5000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_gordon --template_list vol_mist_444 thick_mist_444 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1