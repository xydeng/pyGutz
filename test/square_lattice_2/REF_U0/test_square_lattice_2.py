#!/usr/bin/evn python
#  A TB model interface for Gutz
import numpy as np
from scipy.linalg import block_diag

## import ase module
from ase.dft import kpoints
import sys
sys.path.append("../../")
sys.path.append("/home/ykent/WIEN_GUTZ/bin/")

## import tbBase
from tbASE import *
from tbGutz import *

if __name__=="__main__":
    #Example . SquareLattice.
    ### a normal square lattice. default in gallery
    sTB=TB.gallery().add_spindegeneracy().supercell(extent=(2,2,1))
    kps_size=(5,11,1)
    kps=kpoints.monkhorst_pack(kps_size)
    # unit cell
    ### a Gutz TB model on a square lattice.
    gTB=tbGutz(sTB.Atoms,sTB.Hr,interaction=["Kanamori",(8.0,)])
    gTB.output_CyGutz(kps, num_electrons = 2.4)

    sigma_list = []; U_list = []
    norbitals = sTB.Atoms.nspinorbitals/2
    for i in range(1):
      sig_half = (np.arange(norbitals*norbitals)+1).reshape(norbitals,norbitals)
      sigma_list.append(block_diag(sig_half, sig_half))  # assuming Sz conservation
      U_list.append(np.identity(norbitals*2, dtype = complex))

    from gl_inp import set_wh_hs
    set_wh_hs(sigma_list, U_list)
