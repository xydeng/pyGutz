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
from U_Nambu import *

            
# set up model
# single band, square lattice
## set up Atom and orbitals.
a=AtomsTB("N",[(0,0,0)],cell=(1,1,1))
a.set_orbitals_spindeg()


## set up tight-binding for the spin-unpolarized case. Only interlayer hopping
aTB=TB(a)
aTB.set_hop([((0,1,0),0,0,1),
    ((1,0,0),0,0,1),
    ((0,-1,0),0,0,1),
    ((-1,0,0),0,0,1)])

## Hk: fourier transform
###  ASE kpoints module could be used
Nk=20
kps=kpoints.monkhorst_pack((Nk,Nk,1))
### Hk for the Brillouin zone
Hk=aTB.Hk(kps)

## nambu basis with spin, a constant term of onsite energy is ignored
### 
nambuTB=aTB.add_spindegeneracy().trans_Nambubasis()
kps=kpoints.monkhorst_pack((Nk,Nk,1))
### Hk for the Brillouin zone
Hk_nambu=nambuTB.Hk(kps)

## Umatrix in nambu
spin_names=["up","down"]
orb_names=["s"]
U_matrix=numpy.ones((1,1,1,1))*4.0
#mapop=set_operator_structure(spin_names,orb_names)
if True:
    print "test single band h_loc:"
    hloc=h_loc_slater(spin_names,orb_names,U_matrix,off_diag=True,H_dump="Hloc.dat")
    print hloc
    hloc_nambu=h_loc_slater_nambu(spin_names,orb_names,U_matrix,off_diag=True,H_dump="Hloc_nambu.dat")
    print "in nambu basis:==>"
    print  hloc_nambu
    print ""
