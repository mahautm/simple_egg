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

    for combination in itertools.product(*experiments):
        txt_params = ""
        job_name = "egg"
        for i, element in enumerate(combination):
            txt_params += f" --{param_names[i]} {element}"
            job_name += f"_{param_names[i][:2]}{element}"
        logdir = f"/gpfsscratch/rech/imi/ude64um/simple_egg_exp/{job_name}"
        txt_params += f" --tensorboard_dir={logdir}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        run_exp(job_name, "first_exp.py", txt_params, gpu=True, time="1:00:00")


run_batch("jz/hyper_params.json")
