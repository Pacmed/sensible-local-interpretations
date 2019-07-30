import itertools
from slurmpy import Slurm
import pmlb as dsets

partition = 'high'

# sweep different ways to initialize weights
from dset_names import dset_names
dset_nums = range(0, 94) # len 94
# class_weights = [2, 5, 10, 100]
class_weights = [2]




# run
s = Slurm("pmlb", {"partition": partition, "time": "1-0", "mem": "MaxMemPerNode"})

# iterate
for class_weight in class_weights:
    for i in dset_nums:
        param_str = 'module load python; python3 /accounts/projects/vision/chandan/class-weight-uncertainty/experiments/sweep_pmlb/fit.py '
        param_str += 'dset_name ' + str(dset_names[i]) + ' '
        param_str += 'class_weight ' + str(class_weight)
        s.run(param_str)