import os

prefix = ["vol", "thick"]
MAX_EPOCHS = "6000"
HIDDEN_DIMS = "256 256 256 3"
DROPOUT = "0.5"
LR = "0.0001"
MODEL = "mlp"
WRITE_DIR = "logs"
template_sets = []
cmd_list = []
suffix = "_17"

for i in range(100, 1001, 100):
    schaefer_templates = ""
    for p in prefix:
        schaefer_templates += p + "_schaefer_" + str(i) + suffix + " "
    cmd = "python3 -m " \
          "src.trainers.mlp_trainer --write_dir " + WRITE_DIR + " --exp_name " + MODEL + "_schaefer_"+ str(i) + suffix + \
          " --template_list " + schaefer_templates + "--max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
          " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &"
    print(cmd)
    cmd_list.append(cmd)


cmd_list.append("python3 -m src.trainers.mlp_trainer --write_dir " + WRITE_DIR + " --exp_name " + MODEL + "_aal"+ \
          " --template_list vol_aal thick_aal" + " --max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
          " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &")
cmd_list.append("python3 -m src.trainers.mlp_trainer --write_dir " + WRITE_DIR + " --exp_name " + MODEL + "_aal2"+ \
          " --template_list vol_aal2 thick_aal2" + " --max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
          " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &")
cmd_list.append("python3 -m src.trainers.mlp_trainer --write_dir " + WRITE_DIR + " --exp_name " + MODEL + "_aal3"+ \
          " --template_list vol_aal3 thick_aal3" + " --max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
          " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &")
cmd_list.append("python3 -m src.trainers.mlp_trainer --write_dir " + WRITE_DIR + " --exp_name " + MODEL + "_yeo7"+ \
          " --template_list vol_yeo_7 thick_yeo_7" + " --max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
          " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &")
cmd_list.append("python3 -m src.trainers.mlp_trainer --write_dir " + WRITE_DIR + " --exp_name " + MODEL + "_yeo17"+ \
          " --template_list vol_yeo_17 thick_yeo_17" + " --max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
          " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &")

with open("run.sh", "w") as f:
    f.writelines("#!/bin/bash")
    f.writelines("\n\n")
    for cmd in cmd_list:
        f.writelines(cmd)
        f.writelines("\n")


