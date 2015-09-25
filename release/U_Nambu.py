#import pytriqs.utility.mpi as mpi
from pytriqs.operators import *
from pytriqs.operators.util.op_struct import get_mkind
from U_matrix import *
import numpy


### taken from pytriqs.operator.hamiltonians
### reforms to give the operators in Nambu basis
### this is a bit tricky since one has to refer to the TRIQS.
### 
def h_loc_slater_nambu(spin_names,orb_names,U_matrix,off_diag=None,map_operator_structure=None,H_dump=None):

    if H_dump:
        H_dump_file = open(H_dump,'w')
        H_dump_file.write("Slater Hamiltonian:" + '\n')

    H = Operator()
    mkind = get_mkind(off_diag,map_operator_structure)
    for s1, s2 in product(spin_names,spin_names):
        for a1, a2, a3, a4 in product(orb_names,orb_names,orb_names,orb_names):
            ### a simple reorder here
            U_val = U_matrix[orb_names.index(a1),orb_names.index(a2),orb_names.index(a3),orb_names.index(a4)]
            if abs(U_val.imag) > 1e-10:
                raise RuntimeError("Matrix elements of U are not real. Are you using a cubic basis?")
            
            H_term = 0.5 * U_val.real * op_in_nambubasis(c_dag(*mkind(s1,a1))) *op_in_nambubasis( c_dag(*mkind(s2,a2)))  * op_in_nambubasis(c(*mkind(s2,a4)))  * op_in_nambubasis( c(*mkind(s1,a3)))
            H += H_term

            # Dump terms of H
    if H_dump:
        for i in H:
            print>>H_dump_file,i

    return H

def h_loc_kanamori_nambu(spin_names,orb_names,U,Uprime,J_hund,off_diag=None,map_operator_structure=None,H_dump=None):

    if H_dump:
        H_dump_file = open(H_dump,'w')
        H_dump_file.write("Kanamori Hamiltonian:" + '\n')

    H = Operator()
    mkind = get_mkind(off_diag,map_operator_structure)

    # density terms:
    if H_dump: H_dump_file.write("Density-density terms:" + '\n')
    for s1, s2 in product(spin_names,spin_names):
        for a1, a2 in product(orb_names,orb_names):
            if (s1==s2):
                U_val = U[orb_names.index(a1),orb_names.index(a2)]
            else:
                U_val = Uprime[orb_names.index(a1),orb_names.index(a2)]

            #H_term = 0.5 * U_val * n(*mkind(s1,a1)) * n(*mkind(s2,a2))
            H_term = 0.5 * U_val * op_in_nambubasis(c_dag(*mkind(s1,a1))) *op_in_nambubasis( c_dag(*mkind(s2,a2)))  * op_in_nambubasis(c(*mkind(s2,a2)))  * op_in_nambubasis( c(*mkind(s1,a1)))
            H += H_term

            # Dump terms of H
            if H_dump and not H_term.is_zero():
                H_dump_file.write('%s'%(mkind(s1,a1),) + '\t')
                H_dump_file.write('%s'%(mkind(s2,a2),) + '\t')
                H_dump_file.write(str(U_val) + '\n')

    # spin-flip terms:
    if H_dump: H_dump_file.write("Spin-flip terms:" + '\n')
    for s1, s2 in product(spin_names,spin_names):
        if (s1==s2):
            continue
        for a1, a2 in product(orb_names,orb_names):
            if (a1==a2):
                continue
            H_term = -0.5 * J_hund * op_in_nambubasis(c_dag(*mkind(s1,a1))) * op_in_nambubasis(c(*mkind(s2,a1))) *op_in_nambubasis( c_dag(*mkind(s2,a2))) * op_in_nambubasis(c(*mkind(s1,a2)))
            H += H_term

            # Dump terms of H
            if H_dump and not H_term.is_zero():
                H_dump_file.write('%s'%(mkind(s1,a1),) + '\t')
                H_dump_file.write('%s'%(mkind(s2,a2),) + '\t')
                H_dump_file.write(str(-J_hund) + '\n')

    # pair-hopping terms:
    if H_dump: H_dump_file.write("Pair-hopping terms:" + '\n')
    for s1, s2 in product(spin_names,spin_names):
        if (s1==s2):
            continue
        for a1, a2 in product(orb_names,orb_names):
            if (a1==a2):
                continue
            H_term = 0.5 * J_hund * op_in_nambubasis(c_dag(*mkind(s1,a1))) *op_in_nambubasis( c_dag(*mkind(s2,a1))) * op_in_nambubasis(c(*mkind(s2,a2))) * op_in_nambubasis( c(*mkind(s1,a2)))
            H += H_term

            # Dump terms of H
            if H_dump and not H_term.is_zero():
                H_dump_file.write('%s'%(mkind(s1,a1),) + '\t')
                H_dump_file.write('%s'%(mkind(s2,a2),) + '\t')
                H_dump_file.write(str(-J_hund) + '\n')

    return H

def h_loc_density_nambu(spin_names,orb_names,U,Uprime,off_diag=None,map_operator_structure=None,H_dump=None):

    if H_dump:
        H_dump_file = open(H_dump,'w')
        H_dump_file.write("Density-density Hamiltonian:" + '\n')

    H = Operator()
    mkind = get_mkind(off_diag,map_operator_structure)
    if H_dump: H_dump_file.write("Density-density terms:" + '\n')
    for s1, s2 in product(spin_names,spin_names):
        for a1, a2 in product(orb_names,orb_names):
            if (s1==s2):
                U_val = U[orb_names.index(a1),orb_names.index(a2)]
            else:
                U_val = Uprime[orb_names.index(a1),orb_names.index(a2)]

            H_term = 0.5 * U_val * op_in_nambubasis(c_dag(*mkind(s1,a1))) *op_in_nambubasis( c_dag(*mkind(s2,a2)))  * op_in_nambubasis(c(*mkind(s2,a2)))  * op_in_nambubasis( c(*mkind(s1,a1)))
            H += H_term

            # Dump terms of H
            if H_dump and not H_term.is_zero():
                H_dump_file.write('%s'%(mkind(s1,a1),) + '\t')
                H_dump_file.write('%s'%(mkind(s2,a2),) + '\t')
                H_dump_file.write(str(U_val) + '\n')

    return H

def op_in_nambubasis(op):
    """ transform a cononical fermion operator to nambu basis. (the operator is in real space (H_loc).
        c_dag(*("up","s"))) is a list: [[[[True, ['up', 's']]], 1.0]]
    """
    oplist=list(op)
    dagger=oplist[0][0][0][0]
    spinname=oplist[0][0][0][1][0]
    if spinname=="up":
        newop=op
    elif spinname=="down":
        if dagger:   ## transform c^+ to c
            newop=c(*tuple(oplist[0][0][0][1]))
        else:
            newop=c_dag(*tuple(oplist[0][0][0][1]))
    return newop


if __name__=="__main__":
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


    ##
    if True:
        print "test operators"
        print c_dag(*("up","s")),"==>", op_in_nambubasis(c_dag(*("up","s")))    
        print c(*("up","s")),"==>", op_in_nambubasis(c(*("up","s")))    
        print c_dag(*("down","s")),"==>", op_in_nambubasis(c_dag(*("down","s")))
        print c(*("down","s")),"==>", op_in_nambubasis(c(*("down","s")))
