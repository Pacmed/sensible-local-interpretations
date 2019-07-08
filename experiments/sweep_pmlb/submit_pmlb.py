import itertools
from slurmpy import Slurm
import pmlb as dsets

partition = 'high'

# sweep different ways to initialize weights
from dset_names import dset_names
dset_nums = range(10, 50) # len 94

# run
s = Slurm("pmlb", {"partition": partition, "time": "3-0", "mem": "MaxMemPerNode"})

# iterate
for i in dset_nums:
    param_str = 'module load python; python3 /accounts/projects/vision/chandan/pacmed/experiments/sweep_pmlb/run_pmlb.py dset_name '
    param_str += str(dset_names[i])
    s.run(param_str)