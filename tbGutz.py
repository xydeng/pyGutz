#!/usr/bin/evn python
#  A TB model interface for Gutz
import numpy
import pickle

import glob
import shutil
import os

## import ase module
from ase.dft import kpoints

## U_matrix from triqs.
from U_matrix import *
from pytriqs.operators import *
from pytriqs.operators.util import *

## import tbBase
from tbASE import *
from U_Nambu import *

## import CyGutz interface
import gl_interface


class tbGutz(TB):
    """ tb model with routines for Gutzwiller.

    Attributes:
      Hr: bare hopping
      Hloc_quadratic: quadratic part of local Hamiltonian.
      Hloc_quartic: quatic part of local Hamiltonian.
      h_onsite_noint: a matrix of on-site hopping for all the orbitals, contains noninteraction part only.
      iscorrelated: a list with same structure as orbitals. For example, for a two band system with orbital [("s","p")], iscorrelated= [(0,1)] means the second orbital is correlated. This is for further interface and not used currently.
    """
    def __init__(self,Atoms,Hr=None,h_onsite_noint=None,interaction=None,iscorrelated=None,isnambubasis=False):
        """ Init object. As in TB object, but including also the correlated orbitals

        Args:
          Atoms: AtomsTB objects with structural information, see AtomsTB in tbASE
          Hr: dict of hopping matrix. the bare hopping part is stored in self.Hr.
          h_onsite_noint: (optional) on site energy from noninteracting Hk, if None, h_onsite is taken from Hr, otherwise set by h_onsite_noint.
          interaction: (optional) (interactiontype,interaction parameters). interactiontype="Slater" (F0,F2,F4) or "Kanamori" (U,J) . For the time being, only Kanamori interaction is used.
          iscorrelated: (optional) set flags of correlated orbitals.
          isnambubasis: is in nambu basis, flag for superconducting model. See tests.

        Note h_onsite is a block-diagonalized matrix, which non-zero elements only for the corresponding block for each atom.
        """
        super(tbGutz,self).__init__(Atoms,Hr)
        honsite=self.set_barehopping()
        if h_onsite_noint is None:
            self.set_h_onsite_noint(honsite)
        else:
            self.set_h_onsite_noint(h_onsite_noint)
        ##
        self.set_correlatedorbitals(iscorrelated)
        self.interaction=interaction
        self.isnambubasis=isnambubasis
        self.set_Hloc(isnambubasis)

    def set_barehopping(self):
        """ set the barehopping by removing the onsite energy

        Returns:
          h_onsite: on site energy, which is a list This is to set t_ii_\alpha\beta(R=0) to zero
        """
        R=(0,0,0)
        h_onsite=numpy.zeros((self.Atoms.nspinorbitals,self.Atoms.nspinorbitals),dtype=numpy.complex)
        if R not in self.Hr: ## onsite part is not in the hoppings
            return h_onsite

            # correct only in H[R=0] is diagonal
            #h_onsite[:]=numpy.diag(self.Hr[R])[:]
            #self.Hr[R]-=numpy.diag(h_onsite)

            #
        k=0
        for ispin in xrange(len(self.Atoms.spin)):
            for ia in xrange(len(self.Atoms.orbitals)):  #
                norb=len(self.Atoms.orbitals[ia])
                h_onsite[k:k+norb,k:k+norb]=self.Hr[R][k:k+norb,k:k+norb]
                k+=norb
        self.Hr[R]-=h_onsite

        return h_onsite

    def set_h_onsite_noint(self,h_onsite=None):
        """
        """
        self.h_onsite_noint=h_onsite

    def get_h_onsite_noint(self):
        """
        """
        return self.h_onsite_noint

    def set_correlatedorbitals(self,iscorrelated=None):
        """
        set the attribute iscorrelated.
        """
        if iscorrelated is None:  # by default all orbitals are correlated.
            self.iscorrelated=[tuple([1 for j in i]) for i in self.Atoms.orbitals]
        else:
            self.iscorrelated=iscorrelated

    def get_correlatedorbitals(self):
        return self.iscorrelated

    def solvGutz(self):
        """ to be done. Try to solve the model
        """
        return self.iscorrelated

    def set_Hloc(self,isnambubasis=False):
        """ return interaction part of Hamiltonian
        Args:
          nambubasis: The interaction part is change to Nambu basis.

        Returns:
        H: Comprehensive form of local Hamiltonian.
        H_quadratic: quadratic part of local Hamiltonian. A matrix of same size as Hr.
        H_quartic: a list of all quartic terms, (c^dag,c^dag,c,c, M) form, with M interaction parameters.
        """
        assert self.Atoms.spindeg, "Add spin first!"
        H=Operator()
        if self.interaction[0]=="Slater":
            print "not implemented!!"
            exit()
        elif self.interaction[0]=="Kanamori":
            intvalue=self.interaction[1]
            assert len(intvalue)>=1,"interaction value needs to"
            U,J=0,0
            U=intvalue[0]
            if len(intvalue)>=2:
                J=intvalue[1]

            # for every correlted site get the interaction operators
            for iatom in xrange(len(self.Atoms)):
                if numpy.sum(numpy.array(self.iscorrelated[iatom])) != 0:  # Atom is correlated
                    orbitals=["%s_%s"%(iatom,iorb)  for iorb in xrange(len(self.Atoms.orbitals[iatom])) if self.iscorrelated[iatom][iorb] ]
                    norb=len(orbitals)
                    UU,UUprime=U_matrix_kanamori(norb,U,J)
                    #print self.Atoms.spin, norb,orbitals,UU,UUprime
                    if isnambubasis:
                        H+= h_loc_kanamori_nambu(self.Atoms.spin,orbitals,UU,UUprime,J,off_diag=True)
                    else:
                        H+=h_int_kanamori(self.Atoms.spin,orbitals,UU,UUprime,J,off_diag=True)


        ## add all quadratic part from self.h_onsite_noint
        for iorb in xrange(self.Atoms.nspinorbitals):
            for jorb in xrange(self.Atoms.nspinorbitals):
                t=self.h_onsite_noint[iorb,jorb]
                if abs(t)>1e-8:  # nozero part
                    ispin1,iatom1,iorb1=self.Atoms.idx_spinorbital_sao(iorb)
                    ispin2,iatom2,iorb2=self.Atoms.idx_spinorbital_sao(jorb)
                    cop1=c_dag(self.Atoms.spin[ispin1],"%s_%s"%(iatom1,iorb1))
                    cop2=c(self.Atoms.spin[ispin2],"%s_%s"%(iatom2,iorb2))
                    if isnambubasis:
                        cop1n=op_in_nambubasis(cop1)
                        cop2n=op_in_nambubasis(cop2)
                        H+=cop1n*cop2n*t
                    else:
                        H+=cop1*cop2*t


        ### turn H from comprehensive form to portable form for interface.
        H_quadratic=numpy.zeros((self.Atoms.nspinorbitals,self.Atoms.nspinorbitals),dtype=numpy.complex)  #by default, H_quartic is h_onsite.
        H_quartic=[]
        for iham in H: ## iteration over all terms in H
            oper=iham[0]
            scale=iham[1]
            if len(oper)==2:   # quadratic form
                spin1=oper[0][1][0]
                atom1,orb1=tuple([ int(i) for i in oper[0][1][1].split("_")])
                spin2=oper[1][1][0]
                atom2,orb2=tuple([ int(i) for i in oper[1][1][1].split("_")])
                spinorbital1=self.Atoms.idx_sao_spinorbital[(spin1,atom1,orb1)]
                spinorbital2=self.Atoms.idx_sao_spinorbital[(spin2,atom2,orb2)]
                H_quadratic[spinorbital1,spinorbital2]+=scale
            if len(oper)==4: # quartic part
                spin1=oper[0][1][0]
                #print oper

                atom1,orb1=tuple([ int(i) for i in oper[0][1][1].split("_")])
                spin2=oper[1][1][0]
                atom2,orb2=tuple([ int(i) for i in oper[1][1][1].split("_")])
                spinorbital1=self.Atoms.idx_sao_spinorbital[(spin1,atom1,orb1)]
                spinorbital2=self.Atoms.idx_sao_spinorbital[(spin2,atom2,orb2)]

                spin3=oper[2][1][0]
                atom3,orb3=tuple([ int(i) for i in oper[2][1][1].split("_")])
                spin4=oper[3][1][0]
                atom4,orb4=tuple([ int(i) for i in oper[3][1][1].split("_")])
                spinorbital3=self.Atoms.idx_sao_spinorbital[(spin3,atom3,orb3)]
                spinorbital4=self.Atoms.idx_sao_spinorbital[(spin4,atom4,orb4)]
                H_quartic.append((spinorbital1,spinorbital2,spinorbital3,spinorbital4,scale))
        self.Hloc_comprehensive=H
        self.Hloc_quadratic=H_quadratic
        self.Hloc_quartic=H_quartic
        return H,H_quadratic,H_quartic


    def output_model(self,kps,Hkonly=False,suffix=None):
        """this is for the interface to Gutzwiller codel. structure, on-site
        energy, bare Hk and Umatrix are generated.

        Args:
          kps: kpoints, numpy.ndarry (N,3), in unit of reciprocal lattice vectors.
          Hkonly: bool, if True, only bare Hk will be generated. This is useful when generating band structures.
          suffix: suffix that appends to output file name.

        Outputs: generating three files:
          1. info_model.dat  : cell structure, and on site energy matrix.
          2. Hk.dat          : bare non-interacting Hamiltonian
          3. Hloc.dat        : Hloc for orbitals. with both quadratic and quartic part. Currently Hloca_quadratic contains also uncorrelated orbitals.
        """
        if suffix is None:
            suffix=".dat"
        else:
            suffix+=".dat"
        Hk,kcart=self.Hk(kps)
        # write Hk
        with open("Hk"+suffix,"w") as f:
            f.write("#ktotal, norb\n")
            f.write("%s  %s\n"%(len(kps),self.Atoms.nspinorbitals))
            f.write("# ik  k1 k2 k3 // iorb,jorb, hk[ik,iorb,jorb].real, imag...\n")
            for ik in xrange(len(kps)):
                f.write("%s  %s %s %s\n"%(ik, kps[ik][0],kps[ik][1],kps[ik][2]))
                for iorb in xrange(self.Atoms.nspinorbitals):
                    for jorb in xrange(iorb,self.Atoms.nspinorbitals):
                        th=Hk[ik][iorb,jorb]
                        f.write("%i %i %s %s\n"%(iorb,jorb,th.real,th.imag))
                f.write("\n")
        if not Hkonly: # write also struct information and U_matrix
            with open("info_model"+suffix,"w") as f:
                a=self.Atoms
                natoms=len(a)
                f.write("# primitive cell\n")
                for i in a.cell:
                    f.write("%s  %s  %s\n"%(i[0],i[1],i[2]))
                f.write("# reciprocal cell (in unit of 2pi)\n")
                for i in a.get_reciprocal_cell():
                    f.write("%s  %s  %s\n"%(i[0],i[1],i[2]))
                f.write("# number of atoms\n")
                f.write("%s\n"%len(a))
                f.write("# relative coordinates of each atom\n")
                for i in a.get_scaled_positions():
                    f.write("%s  %s  %s\n"%(i[0],i[1],i[2]))
                f.write("# if is spin polarized \n")
                f.write("%s\n"%a.spindeg)

                f.write("# number of orbitals for each atom, iscorrelated\n")
                for ia in xrange(natoms):
                    f.write("%s  "%len(a.orbitals[ia]))
                    for ior in xrange(len(a.orbitals[ia])):
                        f.write("%s "%self.iscorrelated[ior])
                    f.write("\n")
                f.write("# if is nambu basis\n")
                f.write("%s\n"%self.isnambubasis)

            with open("Hloc"+suffix,"w") as f:
                Hloc_comprehensive=self.Hloc_comprehensive
                Hloc_quadratic=self.Hloc_quadratic
                Hloc_quartic=self.Hloc_quartic
                f.write("#loc hamiltonian.\n")
                f.write("#quadratic part: on site energy matrix, for all orbitals. iorb,jorb, e_real,e_imag\n")
                for iorb in xrange(self.Atoms.nspinorbitals):
                    for jorb in xrange(iorb,self.Atoms.nspinorbitals):
                        th=Hloc_quadratic[iorb,jorb]
                        f.write("%i %i %s %s\n"%(iorb,jorb,th.real,th.imag))
                f.write("#quartic part: (C^dag C^dag C C, scale)\n")
                for ihterm in Hloc_quartic:
                    f.write("%s %s %s %s  %s\n"%(ihterm[0],ihterm[1],ihterm[2],ihterm[3],ihterm[4]))

    def trans_nambubasis(self):
        """ create a new TB with the barehopping to nambu basis. Hloc is set correspondingly.

        This transpose is correct only if the one-site hopping is set to zero.
        Returns:
          A new TB project with modified Hr and Hloc.
        """
        atoms=self.Atoms.copy()
        assert self.Atoms.spin, "Orbital has no spin degeneracy! Add spin degenaracy fisrt! ERROR!"
        assert not self.Atoms.spinorbit, "Can not transform to Nambu within spin-orbit basis! ERROR!"
        atoms.set_orbitals_spindeg(orbitals=self.Atoms.orbitals,spindeg=self.Atoms.spindeg)
        norb=atoms.nspinorbitals
        Hr={}
        for iR in self.Hr:
            Hr[iR]=numpy.zeros((norb,norb),dtype=type(self.Hr[iR][0,0]))
            Hr[iR][0:norb/2,0:norb/2]=self.Hr[iR][0:norb/2,0:norb/2]
            ### for down spin, Hr[R]=-(Hr[-R]).transpose
            minusR=tuple([-i for i in iR])
            assert minusR in self.Hr, "Error!, inverse R is not in the hopping matrix"  # create Hr matrix
            Hr[iR][norb/2:,norb/2:]=-self.Hr[minusR].transpose()[norb/2:,norb/2:]
        # create a new tbGutz orbitals with parameters from the original one.
        tb=tbGutz(atoms,Hr,h_onsite_noint=self.h_onsite_noint,interaction=self.interaction,iscorrelated=self.iscorrelated,isnambubasis=True)
        return tb

    def output_CyGutz(self,kps,num_electrons=None,mpiinfo=None):
        """this is output for the gl_interface to CyGutz codel according to Yao's format.

        Args:
          kps: kpoints, numpy.ndarry (N,3), in unit of reciprocal lattice vectors.
          num_electrons: number of electrons
          mpiinfo: (nprocs, master), by default, nprocs=1,master=0

        Outputs: generating three files:
          1. GUTZ1.INP       :
          2. GUTZ2.INP       :
          3. GUTZ3.INP       :
          4. GUTZ3.INP       :
          5. GUTZ3.INP       :
        """
        ## GMPI_?.INP
        if mpiinfo is None:
            nprocs=1
            master=0
        else:
            nprocs=mpiinfo[0]
            master=mpiinfo[1]
        for myrank in xrange(nprocs):
            gl_interface.write_gl_mpi(myrank,nprocs,master)

        ##
        num_atoms=len(self.Atoms)
        ## Gutz1.INP
        units=0
        gl_interface.write_gutz1(num_atoms,units)

        # Gutz2.INP
        num_kpts=len(kps)
        index_spin_orbit= 2 if self.Atoms.spinorbit else 1
        index_spin_bare = 2 if (self.Atoms.spindeg and not self.Atoms.spinorbit) else 1
        # tricky thing, if spin-up and spin-down  re not supposed to be treated sepaarted, we set index_spin_orbit=2
        if index_spin_bare == 2: index_spin_orbit= 2

        max_num_bands=self.Atoms.nspinorbitals
        gl_interface.write_gutz2(index_spin_orbit,index_spin_bare,max_num_bands,num_kpts)

        # Gutz3.INP
        gl_interface.write_gutz3(0,unitary_trans=None,translations=None)

        # Gutz4.INP
        ### a bit tricky here since in principle cluster has to be defined.
        #So here I force here that all the correlated atom in the unit cell form a cluster and correspond to different orbitals, that is, the num_corr_atoms is always 1.
        num_corr_atoms=1
        max_dim_sorbit=self.Atoms.nspinorbitals
        ## not max_dim_sorbit consider spin index only when spinorbit is not considered.
        #if (not self.Atoms.spinorbit) and self.Atoms.spindeg:
        #    max_dim_sorbit/=2
        U_CH_to_local_basis=None
        gl_interface.write_gutz4(max_dim_sorbit,num_corr_atoms,U_CH_to_local_basis)

        ## Gutz5.INP
        weight_kpts=1.0/num_kpts*numpy.ones((num_kpts))
        ##
        if not self.Atoms.spindeg:
            weight_kpts*=2.0
        index_smear=0
        delta=1e-2
        if num_electrons is None: # by default, half filled.
            num_electrons=self.Atoms.nspinorbitals*1.0/len(self.Atoms.spin)
        index_bands=numpy.zeros((num_kpts,3),dtype=numpy.int)
        index_bands[:,0]=self.Atoms.nspinorbitals
        index_bands[:,1]=1
        index_bands[:,2]=self.Atoms.nspinorbitals
        gl_interface.write_gutz5(num_kpts,weight_kpts,index_smear,delta,num_electrons,index_bands)

        ### BNDU_
        # define and diganolize Hk
        Norb=self.Atoms.nspinorbitals
        Hloc_quadratic=self.Hloc_quadratic
        for myrank in xrange(nprocs):
            nkt=(len(kps)+nprocs-1) // nprocs   # get a ceiling division
            k_list=[i for i in xrange(nkt*myrank,min(nkt*myrank+nkt,len(kps)))]
            ek_list=numpy.zeros((len(k_list),Norb),dtype=numpy.float)
            Uk_list=numpy.zeros((len(k_list),Norb,Norb),dtype=numpy.complex)
            for ik in k_list:
                hk,ikpscart=self.Hk([kps[ik]])
                ## add h_on_site to hk
                ekt,Ukt=numpy.linalg.eigh(hk[0]+Hloc_quadratic)   # add Hloc_quadratic
                ### sort ekt Ukt
                sorta=ekt.real.argsort()
                ekt_sorted=ekt[sorta].real
                Ukt_sorted=Ukt[:,sorta]
                ek_list[ik]=ekt_sorted
                Uk_list[ik]=Ukt_sorted.transpose().conjugate()   # get the complex conjugate
                # check eigen vector is correctly generated
                #print numpy.abs(numpy.dot(hk[0]+Hloc_quadratic,Ukt_sorted[:,0])-ekt_sorted[0]*Ukt_sorted[:,0])

            gl_interface.write_gutz_bndu(myrank,ek_list,Uk_list)

        ##Couloumb U
        Norb=self.Atoms.nspinorbitals
        U_matrix_full_list=numpy.zeros((Norb,Norb,Norb,Norb))
        Hloc_quartic=self.Hloc_quartic
        ### this should be correct, however should be checked. Note what I have is the local Hamiltonian operator form explicitly. This should be more compact than the Umatrix itself, which contains redundancy. So reconver U_matrix, in principle I should carefully distinguish the spin index and the order of operators. I would avoid to do that, and simply choose a U_matrix to recover the correct local Hamiltonian.
        for ihterm in Hloc_quartic:
            U_matrix_full_list[(ihterm[0],ihterm[1],ihterm[3],ihterm[2])]=ihterm[4]
        gl_interface.write_coulomb_full([U_matrix_full_list,])


if __name__=="__main__":
    #Example . SquareLattice.
    ### a normal square lattice. default in gallery
    aTB=TB.gallery().add_spindegeneracy()

    # unit cell
    ### a Gutz TB model on a square lattice.
    gTB=tbGutz(aTB.Atoms,aTB.Hr,interaction=["Kanamori",(4.0,)])
    ###print gTB.get_Hloc()[0]
    kps_size=(10,10,1)
    kps=kpoints.monkhorst_pack(kps_size)
    gTB.output_model(kps,suffix="_pcell")
    gTB.output_CyGutz(kps)
    ### dos
    #gTB.get_dos((400,400,1),saveto="gutz_pcell_dos.dat")

    ### nambu basis from an tbGutz object
    #nTB=gTB.trans_nambubasis()
    #nTB.output_model(kps,suffix="_pcell_nambu")
    #nTB.get_dos((400,400,1),saveto="gutz_pcell_dos_n.dat")

    #supercell 2x2

    ### sTB=TB.gallery().supercell(extent=(2,2,1)).add_spindegeneracy()
    # same as the line above.
    #sTB=TB.gallery().add_spindegeneracy().supercell(extent=(2,2,1))
    #### unit cell
    ### a Gutz TB model on a square lattice.
    #gTB=tbGutz(sTB.Atoms,sTB.Hr,interaction=["Kanamori",(4.0,)])
    #print gTB.get_Hloc()[0]
    #kps_size=(10,10,1)
    #kps=kpoints.monkhorst_pack(kps_size)
    #gTB.output_model(kps,suffix="_scell2x2")
    #gTB.get_dos((400,400,1),saveto="gutz_dos_scell2x2.dat")

    ### nambu basis from an tbGutz object
    #nTB=gTB.trans_nambubasis()
    #nTB.output_model(kps,suffix="_scell2x2_nambu")
    #nTB.get_dos((400,400,1),saveto="gutz_dos_scell2x2_n.dat")
