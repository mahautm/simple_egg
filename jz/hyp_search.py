# run experiments on a cluster (here jeanzay)

import json
import itertools
from looper import run_exp
import os


def run_batch(path):
    with open(path) as jsonFile:
        grid = json.load(jsonFile)
        jsonFile.close()
    experiments = []
    param_names = []
    for name in grid:
        experiments.append(grid[name])
        param_names.append(name)
    count = 0
    for combination in itertools.product(*experiments):
        txt_params = ""
        job_name = "egg"
        for i, element in enumerate(combination):
            txt_params += f" --{param_names[i]} {element}"
            if "data" not in param_names[i]:  # avoid paths
                job_name += f"_{param_names[i][:2]}{element}"
        logdir = f"/gpfsscratch/rech/imi/ude64um/simple_egg_exp/{job_name}"
        txt_params += f" --tensorboard_dir={logdir}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        # print(txt_params)
        run_exp(job_name, "first_exp.py", txt_params, gpu=True, time="5:00:00")
        count += 1
    print(f"submitted {count} jobs")


run_batch("jz/hyper_params.json")
