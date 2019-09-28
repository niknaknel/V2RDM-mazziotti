from functools import reduce
import numpy, pyscf, rdm_tools, itertools, time
import makeSDPAB
from pyscf import gto, scf, mcscf, ao2mo, fci

class testSuite():

    def twoElectronAB(basis, flg):

        mol = gto.Mole()
        mol.atom = [['H', (0., 0., 0.)], ['H', (0.,0.,0.5)], ['H', (0.,0.5,0.)]]
        #mol.atom = [['H', (0.,0.,0.)], ['H', (0.,0.,0.5)]]
        #mol.atom = [['H', (0.,0.,0.,)], ['F', (0.,0.,0.9)]]
        #mol.atom = [['H', (0.,0.,0.,)], ['B', (0.,0.,1.2325)]]
        #mol.atom = [['H', (0.,0.,0.,)], ['Li', (0.,0.,5.)]]
        #mol.atom = [['H', (0.,0.,0.,)], ['Li', (0.,0.,1.5949)]]
        #mol.atom = [['H', (0.,0.,0.,)], ['B', (0.,0.,2.5)]]
        mol.atom = [['Be', (0.,0.,0.)]]
        #mol.atom = [['H', (0.,0.,0.)]]
        mol.spin = 0
        mol.basis = 'STO-3G'#'6-31g'
        #mol.charge = -1
        mol.verbose =0
        mol.build()
        mf = scf.RHF(mol)
        mf.scf()
        r = mol.nao_nr()
        Na, Nb = mol.nelec
        mc = mcscf.CASCI(mf, mol.nao_nr(), (Na,Nb))
        mc.verbose = 3
        #mc.fcisolver = fci.solver(mol, singlet=True)
        mc.kernel()
        (d1a, d1b), (d2aa, d2ab, d2bb) = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas, (Na, Nb))
        v2aa = ao2mo.kernel(mol, mf.mo_coeff, compact=False)
        v2aa = v2aa.reshape(r,r,r,r).swapaxes(1,2)
        k1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        print('Nuclear Rep', mol.energy_nuc())
        N = mol.nelectron
        Ns = N//2
        t0 = time.time()
        tF = time.time()
        print('Time to prep Hamiltonian ', str(round(tF-t0, 5)), ' seconds')
        if flg == 'sdp':
            # return SDP problem
            k2 = makeK2(v2aa.reshape(r,r,r,r), k1, r, N)
            k2a = asym_K2(k2, r)
            k2a = compactK2(k2a, r)
            c = numpy.append(k2a.flatten(), k2.flatten())
            c = numpy.append(c, k2a.flatten())
            t0 = time.time()
            #sdp = makeSDPAB.makeSDPConstraints((['D2aa', 'D2ab', 'D2bb', 'D1a', 'D1b', 'Q1a', 'Q1b', 'Q2aa', 'Q2ab', 'Q2bb', 'G2ab', 'G2ba', 'G2big'], mol.nao_nr(), (Na, Nb)))
            sdp = makeSDPAB.makeSDPConstraints(('DQG', mol.nao_nr(), (Na,Nb)))
            tF = time.time()
            print('Time to make Constraints ', str(round(tF-t0, 5)), ' seconds')
            sdp.makeAB()
            t1 = time.time()
            print('Time to make SDP ', str(round(t1-tF, 5)), ' seconds')
            pad_c = numpy.zeros((sdp.numVars - len(c)))
            c = numpy.append(c, pad_c)
            A, b, blocks = sdp.A, sdp.b, sdp.blocks

        return A, b, c, blocks, sdp.varlist

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
