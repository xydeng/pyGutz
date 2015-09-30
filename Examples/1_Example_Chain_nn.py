#!/usr/bin/evn python
#  A TB model interface for Gutz
import numpy
import pickle
from scipy.linalg import block_diag

import glob
import shutil
import os,sys

### include  in loadPATH the WIEN_GUTZ script. It would be better to borrow some script from there
sys.path.append("/home/xiaoyu/Public/Soft//WIEN_GUTZ/Wien_Gutz_4.5/bin/")
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
    if True: #DOS
        gTB=tbGutz(aTB.Atoms,aTB.Hr,interaction=["Kanamori",(1.0,)])
        #check density of states
        gTB.get_dos(kps_size,saveto="chain_dos_cell.dat")
        gTB.output_CyGutz(kps)

    ### write WH_HS.INP
        sigma_list = []; U_list = []
        norbitals = gTB.Atoms.nspinorbitals/2
        for i in range(1):
            sig_half = (numpy.arange(norbitals*norbitals)+1).reshape(norbitals,norbitals)
            sigma_list.append(block_diag(sig_half, sig_half))  # assuming Sz conservation
            U_list.append(numpy.identity(norbitals*2, dtype = complex))

        from gl_inp import set_wh_hs, set_gl_inp
        set_wh_hs(sigma_list, U_list)
    #### write GL.INP
        # GL.INP
        spin_pol = 'n'
        SOC = ['y']; CF = ['y']
        NTYPE = 1; NIONS = 1; ITYPE_list = [1]
        NI0_list = [1]; NIMAP_list = [1]; corr_atom_type = ["X"]
        type_1atom = [0]; df_list = ["g"]; dim_list = [norbitals*2]
        log = open("init_ga_a.slog", 'w')
        set_gl_inp(spin_pol, SOC, CF, NTYPE, NIONS, ITYPE_list, NI0_list, NIMAP_list, corr_atom_type, type_1atom, df_list, dim_list, log)
        log.close()

    #supercell 2x1
    if False:
        #### super cell
        gTB=tbGutz(sTB.Atoms,sTB.Hr,interaction=["Kanamori",(1.0,)])
        gTB.get_dos(kps_size,saveto="chain_dos_scell.dat")
        gTB.output_CyGutz(kps)

        ### write WH_HS.INP
        sigma_list = []; U_list = []
        norbitals = gTB.Atoms.nspinorbitals/2
        for i in range(1):
            sig_half = (numpy.arange(norbitals*norbitals)+1).reshape(norbitals,norbitals)
            sigma_list.append(block_diag(sig_half, sig_half))  # assuming Sz conservation
            U_list.append(numpy.identity(norbitals*2, dtype = complex))

        from gl_inp import set_wh_hs, set_gl_inp
        set_wh_hs(sigma_list, U_list)
    #### write GL.INP
        # GL.INP
        spin_pol = 'n'
        SOC = ['y']; CF = ['y']
        NTYPE = 1; NIONS = 1; ITYPE_list = [1]
        NI0_list = [1]; NIMAP_list = [1]; corr_atom_type = ["X"]
        type_1atom = [0]; df_list = ["g"]; dim_list = [norbitals*2]
        log = open("init_ga_s.slog", 'w')
        set_gl_inp(spin_pol, SOC, CF, NTYPE, NIONS, ITYPE_list, NI0_list, NIMAP_list, corr_atom_type, type_1atom, df_list, dim_list, log)
        log.close()

