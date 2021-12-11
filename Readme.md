## Final Project: Classification of different stages of Alzheimer's disease
This repository contains code for the final project of the course BIOS 772.
There are 3 different models: MLP, MM-MLP, and, Ensemble classifier.

### Environment Setup

Using Conda:
```
conda create --name bios772_env
conda activate bios772_env
pip install -r requirements.txt
```

### Training
MLP model:
```
python3 -m src.trainers.mlp_trainer --write_dir logs --exp_name mlp_mist_444 --template_list vol_mist_444 thick_mist_444 --max_epochs 6000 -l 256 256 256 3 -d 0.5 --lr 0.0001 --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1
```
MM-MLP model:
```
python3 -m src.trainers.multi_modal_trainer --enc_hidden_layers 32 32 --latent_dim 32 -b 32 -g 1 -m 10000 --cls_layers 256 256 3 -d 0.5 --lr 0.0001 --mode train --batch_norm 1 --template_list vol_mist_444 thick_mist_444 vol_schaefer_200_7 thick_schaefer_200_7 --write_dir logs --exp_name multi_modal_cls_mist_444_schaefer_200
```
Ensemble model:
```
python3 -m src.trainers.ensemble_trainer --write_dir logs --pair_name NC_AD --exp_name ensemble_NC_AD_mist_444 --template_list vol_mist_444 thick_mist_444 --max_epochs 10000 -l 256 256 256 2 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1
python3 -m src.trainers.ensemble_trainer --write_dir logs --pair_name NC_MCI --exp_name ensemble_NC_AD_mist_444 --template_list vol_mist_444 thick_mist_444 --max_epochs 10000 -l 256 256 256 2 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1
python3 -m src.trainers.ensemble_trainer --write_dir logs --pair_name MCI_AD --exp_name ensemble_MCI_AD_mist_444 --template_list vol_mist_444 thick_mist_444 --max_epochs 10000 -l 256 256 256 2 -d 0.5 --lr 0.0001 --mode train -g 0 --split 0.8 --batch_size 32 --batch_norm 1
```

Important Arguments:
* --template_lists: Choose one or more atlases by specifying template list from utils/data_utils/template_to_idx_mapping
* -g : whether to use a gpu or not
* --write_dir: The output directory where checkpoints and tensorboard log dir will be written
* --exp_name: name of the experiment
* --load_from_ckpt: whether to start training from a checkpoint
* --ckpt: checkpoint file name if load_from_ckpt is true.
### Predictions
```
python3 -m src.predictions.predict_mlp
python3 -m src.predictions.predict_mm_mlp
python3 -m src.predictions.predict_ensemble
```

### Final predictions
Final prediction results are saved in the prediction_labels folder: 
- prediction_labels/ensemble_prediction_labels.csv
- prediction_labels/mlp_prediction_labels.csv
- prediction_labels/mm-mlp_prediction_labels.csv