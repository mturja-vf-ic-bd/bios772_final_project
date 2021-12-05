import os
from pathlib import Path


def find_last_saved_version(dir, prefix="run"):
    dirs = [f for f in os.listdir(dir) if f.startswith(prefix)]
    if len(dirs) == 0:
        return 0
    run_ids = [int(id[len(prefix)+1:]) for id in dirs]
    run_ids.sort()
    return run_ids[-1] + 1


def create_version_dir(dir, prefix="run"):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True)
    id = find_last_saved_version(dir, prefix)
    next_id = prefix + "_" + str(id)
    version_folder = os.path.join(dir, next_id)
    os.mkdir(version_folder)
    return version_folder
