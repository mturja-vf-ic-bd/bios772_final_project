#!/bin/bash


python3 -m src.trainers.mlp_trainer --template_list vol_schaefer_200_7 --write_dir mlp_logs --max_epochs 1000  -l 256 256 256 3 -d 0.05 --lr 0.001 --mode train -g 1 --exp_name mlp --split 0.8 --batch_size 32 --dropout 0.5

ssh -N -f -L localhost:20010:localhost:20010 mturja@theia.ia.unc.edu
ssh -N -f -L localhost:6006:localhost:20010 mturja@janus.ia.unc.edu