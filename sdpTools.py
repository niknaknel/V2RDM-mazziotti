from functools import reduce
import numpy, pyscf, rdm_tools, itertools, time
import makeSDPAB
from pyscf import gto, scf, mcscf, ao2mo, fci


def twoElectronAB(geometry, basis):

    mol = gto.Mole()
    # mol.atom = [['H', (0., 0., 0.)], ['H', (0.,0.,0.5)], ['H', (0.,0.5,0.)]]
    # mol.atom = [['H', (0.,0.,0.)], ['H', (0.,0.,0.5)]]
    # mol.atom = [['H', (0.,0.,0.,)], ['F', (0.,0.,0.9)]]
    # mol.atom = [['H', (0.,0.,0.,)], ['B', (0.,0.,1.2325)]]
    # mol.atom = [['H', (0.,0.,0.,)], ['Li', (0.,0.,5.)]]
    # mol.atom = [['H', (0.,0.,0.,)], ['Li', (0.,0.,1.5949)]]
    # mol.atom = [['H', (0.,0.,0.,)], ['B', (0.,0.,2.5)]]
    # mol.atom = [['Be', (0.,0.,0.)]]
    # mol.atom = [['H', (0.,0.,0.)]]

    mol.basis = basis
    mol.atom = geometry
    mol.spin = 0
    #mol.charge = -1
    mol.verbose = 0
    mol.build()

    # run hartree-fock
    mf = scf.RHF(mol)
    mf.scf()

    # number of spin-orbitals
    r = mol.nao_nr()

    # number of electrons
    Na, Nb = mol.nelec # (alpha/beta)
    N = mol.nelectron

    # run casci
    mc = mcscf.CASCI(mf, mol.nao_nr(), (Na,Nb))
    mc.kernel()
    mc.verbose = 3
    mc.fcisolver = fci.solver(mol, singlet=True)
    (d1a, d1b), (d2aa, d2ab_fci, d2bb) = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas, (Na, Nb))

    # get nuclear repulsion
    nuclear_repulsion = mol.energy_nuc()

    # get one body integrals
    k1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))

    # get two body integrals
    v2aa = ao2mo.kernel(mol, mf.mo_coeff, compact=False)
    v2aa = v2aa.reshape(r,r,r,r).swapaxes(1,2)

    # make reduced Hamiltonian
    k2 = makeK2(v2aa, k1, r, N)
    k2a = asym_K2(k2, r)
    k2a = compactK2(k2a, r)
    c = numpy.append(k2a.flatten(), k2.flatten())
    c = numpy.append(c, k2a.flatten())

    # make constraints
    #sdp = makeSDPAB.makeSDPConstraints((['D2aa', 'D2ab', 'D2bb', 'D1a', 'D1b', 'Q1a', 'Q1b', 'Q2aa', 'Q2ab', 'Q2bb', 'G2ab', 'G2ba', 'G2big'], mol.nao_nr(), (Na, Nb)))
    sdp = makeSDPAB.makeSDPConstraints(('DQG', mol.nao_nr(), (Na,Nb)))

    # make SDP
    sdp.makeAB()
    pad_c = numpy.zeros((sdp.numVars - len(c)))
    c = numpy.append(c, pad_c)
    A, b, blocks = sdp.A, sdp.b, sdp.blocks

    # reshape exact D2
    d2ab_fci = d2ab_fci.transpose(0, 2, 3, 1).reshape(r ** 2, r ** 2)

    return A, b, c, blocks, sdp.varlist, nuclear_repulsion, k2, d2ab_fci


def makeK2(v2, k1, r, N):
    h2e = numpy.zeros((v2.shape))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                for l in range(r):

                    jl = 1. if j == l else 0.
                    ik = 1. if i == k else 0.

                    h2e[i,j,k,l] = (k1[j,l]*ik + k1[i,k]*jl)/(N-1.) + v2[i,j,k,l]
    return h2e.reshape(r**2, -1)


def getSSq(d2ab, r):
    # Calculation S squared
    sz = 0.
    for i in range(r):
        for j in range(r):
            bra = i*r+j
            ket = j*r+i
            ind = r*(i+j) + i + j
            sz += d2ab[bra,ket]
            #sz += d2ab.flatten()[ind]
    return sz


def asym_K2(k2, r):
    k2 = k2.reshape(r,r,r,r)
    k2a = numpy.zeros((k2.shape))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                for l in range(r):
                    k2a[i,j,k,l] = k2[i,j,k,l] - k2[i,j,l,k]
    return k2a.reshape(r**2, -1)


def compactK2(k2, r):
        k2 = k2.reshape(r**2, -1)
        rA = r*(r-1)//2
        k2a = numpy.zeros((rA, rA))
        Asymbas = list(itertools.combinations(range(r), 2))
        symbas = list(itertools.product(range(r), repeat=2))

        for bra in Asymbas:
                for ket in Asymbas:
                        k2a[Asymbas.index(bra), Asymbas.index(ket)] = k2[symbas.index(bra), symbas.index(ket)]
        return k2a
