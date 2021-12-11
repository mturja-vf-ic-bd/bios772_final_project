
prefix = ["vol", "thick"]
MAX_EPOCHS = "5000"
HIDDEN_DIMS = "256 256 256 2"
DROPOUT = "0.5"
LR = "0.0001"
MODEL = "ensemble"
WRITE_DIR = "logs"
template_sets = []
cmd_list = []
suffix = "_7"

for pair_name in ["NC_AD", "NC_MCI", "MCI_AD"]:
    for i in range(100, 1001, 100):
        schaefer_templates = ""
        for p in prefix:
            schaefer_templates += p + "_schaefer_" + str(i) + suffix + " "
        cmd = "python3 -m " \
              "src.trainers.ensemble_trainer --write_dir " + WRITE_DIR + " --pair_name " + pair_name + " --exp_name " + MODEL + "_" + pair_name + "_schaefer_"+ str(i) + suffix + \
              " --template_list " + schaefer_templates + "--max_epochs " + MAX_EPOCHS + " -l " + HIDDEN_DIMS + " -d " + DROPOUT + " --lr " + LR + \
              " --mode train -g 1 --split 0.8 --batch_size 32 --batch_norm 1 &"
        print(cmd)
        cmd_list.append(cmd)


with open("run_ensemble.sh", "w") as f:
    f.writelines("#!/bin/bash")
    f.writelines("\n\n")
    for cmd in cmd_list:
        f.writelines(cmd)
        f.writelines("\n")


