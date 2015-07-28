#!/usr/bin/evn python
#  A TB model interface for Gutz
import numpy
import pickle

import glob
import shutil
import os

## import ase module
from ase.dft import kpoints
import sys
sys.path.append("../../")

## import tbBase
from tbASE import *
from tbGutz import *

if __name__=="__main__":
    #Example . SquareLattice.
    ### a normal square lattice. default in gallery
    aTB=TB.gallery().add_spindegeneracy()
    sTB=TB.gallery().add_spindegeneracy().supercell(extent=(1,1,1))
    kps_size=(11,11,1)
    kps=kpoints.monkhorst_pack(kps_size)
    # unit cell
    ### a Gutz TB model on a square lattice.
    gTB=tbGutz(aTB.Atoms,aTB.Hr,interaction=["Kanamori",(8.0,)])
    gTB.output_CyGutz(kps, num_electrons = 0.6)
