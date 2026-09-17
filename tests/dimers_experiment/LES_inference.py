# =========================
# 1. Imports and setup
# =========================
import os
import numpy as np
import torch
#import argparse
import pathlib
from main import get_newest_rundir

# ---- GPU configuration ----
# Force use of GPU 0 (adjust if needed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print('CUDA available =', cuda_available)

num_devices = torch.cuda.device_count()
print('# CUDA devices =', num_devices)


# =========================
# 2. ASE imports
# =========================

from ase.io import read, write
from ase.md.langevin import Langevin

# =========================
# 3. Calculator imports
# =========================
from franken.rf.model import FrankenPotential
from franken.rf.les_model import LESFrankenPotential
from franken.calculators import FrankenCalculator

# =========================
# 4. Define calculator
# =========================
dataset_id=0 #change manually
franken_dir = get_newest_rundir(pathlib.Path(f"les_outputs/dimer_{dataset_id}"))
calc  = FrankenCalculator(
    franken_dir / "best_ckpt.pt",
    model_class=LESFrankenPotential,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)


# =========================
# 6. Read data
# =========================
input_file_path=f"dimer_{dataset_id}_dataset/"
input_file="train.xyz"
dataset=read(input_file_path+input_file,index=':',format='extxyz')

for snap in dataset:
    
    # Overwrite stored results
    if "energy" in snap.info:
        del snap.info["energy"]
    if "forces" in snap.arrays:
        del snap.arrays["forces"]
    if "momenta" in snap.arrays:
        del snap.arrays["momenta"]
    
    #Remove any constraints that can alter forces
    del snap.constraints

    snap.calc = calc

    # trigger calculation
    calc.calculate(
        snap,
        properties=["energy", "forces", "LES_charges"]
    )
    print(snap.calc.results.keys())
    
    #snap.set_calculator(calc)
    #snap.info["energy"]=snap.get_potential_energy()   
    #snap.arrays["forces"]=snap.get_forces()
    snap.arrays["initial_charges"]=snap.calc.results["LES_charges"]
    
    output_file=franken_dir / f"predicted_{input_file}"
    write(output_file,snap,format='extxyz',append=True)
