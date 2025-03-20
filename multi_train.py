#!/usr/bin/env python3

import sys
import subprocess
import multiprocessing

import numpy as np

# from run_canon import coinrun_dirs, cartpole_dirs


def run_script(_):
    """
    Run hyperparameter_optimization.py using the Python interpreter
    from the specified virtual environment.
    """
    subprocess.run(["/vol/bitbucket/tfb115/EfficientIRL/venv/bin/python3", "hyperparameter_optimization.py"])

def run_canon(_, model_files, tag):
    """
    Run hyperparameter_optimization.py using the Python interpreter
    from the specified virtual environment.
    """
    base_dir = "/vol/bitbucket/tfb115/goal-misgen/"
    model_files = [base_dir + m for m in model_files]
    # subprocess.run(["/vol/bitbucket/tfb115/goal-misgen/opvenv/bin/python3.8", f"multi_canon.py --model_files {' '.join(model_files)} --tag {tag}"])
    cmd = ["/vol/bitbucket/tfb115/goal-misgen/opvenv/bin/python3.8",
           "multi_canon_local.py",
           "--model_files",
           *model_files,
           "--tag",
           tag]
    subprocess.run(cmd)

def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_train.py <n>")
        print("Example: python multi_train.py 5")
        sys.exit(1)

    n = int(sys.argv[1])  # Number of parallel processes

    # tag = "hp0"
    # model_files = [x.tolist() for x in np.array_split(cartpole_dirs, n)]
    args = [i for i in range(n)]
    # Verify that the venv's python exists

    with multiprocessing.Pool(processes=n) as pool:
        # Distribute the same venv_python path to each parallel worker
        pool.starmap(run_script, args)


if __name__ == "__main__":
    main()
