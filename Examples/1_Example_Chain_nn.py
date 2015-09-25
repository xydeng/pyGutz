#!/usr/bin/evn python
#  A TB model interface for Gutz
import numpy
import pickle

import glob
import shutil
import os

## import ase module
from ase.dft import kpoints
## import tbBase
from tbASE import *
from tbGutz import *

if __name__=="__main__":
    #Example. A Chain with nearest neighbour hopping
    aTB=TB.gallery("Chain_nn").add_spindegeneracy()
    sTB=TB.gallery("Chain_nn").add_spindegeneracy().supercell(extent=(2,1,1))
    kps_size=(4000,1,1)
    kps=kpoints.monkhorst_pack(kps_size)
    # unit cell
    ### a TB model on a Chain
    if False: #DOS
        gTB=tbGutz(aTB.Atoms,aTB.Hr,interaction=["Kanamori",(4.0,)])
        #check density of states
        gTB.get_dos(kps_size,saveto="chain_dos_cell.dat")
        gTB.output_CyGutz(kps)

    #supercell 2x1
    if True:
        #### super cell
        gTB=tbGutz(sTB.Atoms,sTB.Hr,interaction=["Kanamori",(4.0,)])
        gTB.get_dos(kps_size,saveto="chain_dos_scell.dat")
        gTB.output_CyGutz(kps)
