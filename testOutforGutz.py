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
    #Example . SquareLattice.
    ### a normal square lattice. default in gallery
    aTB=TB.gallery().add_spindegeneracy()
    sTB=TB.gallery().add_spindegeneracy().supercell(extent=(2,2,1))
    kps_size=(10,10,1)
    kps=kpoints.monkhorst_pack(kps_size)
    # unit cell
    ### a Gutz TB model on a square lattice.
    if True:
        gTB=tbGutz(aTB.Atoms,aTB.Hr,interaction=["Kanamori",(4.0,)])
        gTB.output_CyGutz(kps)

    if False:
        gTB=tbGutz(aTB.Atoms,aTB.Hr,interaction=["Kanamori",(4.0,)])            ### nambu basis from an tbGutz object
        nTB=gTB.trans_nambubasis()
        nTB.output_CyGutz(kps)



    #supercell 2x2
    if False:
        #### unit cell
        ### a Gutz TB model on a square lattice.
        gTB=tbGutz(sTB.Atoms,sTB.Hr,interaction=["Kanamori",(4.0,)])
        print gTB.Atoms.nspinorbitals
        gTB.output_CyGutz(kps)

    if False:
    ### nambu basis from an tbGutz object
        gTB=tbGutz(sTB.Atoms,sTB.Hr,interaction=["Kanamori",(4.0,)])
        nTB=gTB.trans_nambubasis()
        nTB.output_CyGutz(kps)


