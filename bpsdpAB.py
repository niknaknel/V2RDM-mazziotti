import sys
import scipy, numpy, time, rdm_tools
from scipy import sparse
from scipy.sparse import linalg
from functools import reduce
import numpy as np
numpy.set_printoptions(linewidth=10000, threshold=100000, precision=3)

import sdpTools
from molecules import OPTIONS

class boundaryPointSDP():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.A = None
        self.b = None
        self.c = None
        self.AAt = None
        self.precond = None
        self.blocks = None
        self.numVars = None
        self.numconst = None
        self.updateMu = 500
        self.mu = 0.6
        self.tau = 1.6
        self.maxIter = 1000
        self.tol = 1e-4
        self.eTol = 1e-6
        self.conv = False

        # timing variables
        self.tCG = 0.
        self.tUM = 0.
        self.tTot = 0.

    def primalError(self):
        A = self.A
        x = self.x
        b = self.b

        # $ |Ax-b| $
        return numpy.linalg.norm(numpy.dot(A,x) - b)

    def dualError(self):
        A = self.A
        y = self.y
        c = self.c
        z = self.z

        # $ |A^T y -c + z|
        return numpy.linalg.norm(numpy.dot(A.T, y) - c + z)

    def dualityGap(self):
        return numpy.inner(self.x, self.z)

    def primalEnergy(self):
        return numpy.dot(self.c.T, self.x)

    def dualEnergy(self):
        return numpy.dot(self.b.T, self.y)

    def solveSDP(self):
        def yFromCG(self):
            '''
            Solve conjugate gradient problem using scipy for now

            sA y = sB
            AA.T y = A(c-z) + tau*mu*(b - Ax)
            '''

            A = self.A
            x = self.x
            b = self.b
            c = self.c
            z = self.z
            mu = self.mu
            tau = self.tau

            sA = self.AAt
            sB = numpy.dot(A, (c-z)) + tau*mu*(b - numpy.dot(A,x))

            cgRes = scipy.sparse.linalg.cg(sA, sB, maxiter=10000, M=self.precond, atol=1e-9)

            if cgRes[1] == 0:
                return cgRes[0]

            else:
                print('\nConjugate Gradient not Converged! Quitting BPSDP.\n')
                return None

        def updateMx(self):
            '''
            Update M(x) matrix

            U = VLV.T

            Up = V L[L>0] V.T
            Un = V L[L<0] V.T

            U = Up + Un

            x = Up/mu
            z = Un*-1.
            '''

            A = self.A
            x = self.x
            y = self.y
            c = self.c
            blocks = self.blocks
            mu = self.mu

            U = mu*x + numpy.dot(A.T, y) - c

            startInd = 0
            for r in blocks:

                numVars = r**2
                # Grab appropriate block of x and make a matrix
                mat = U[startInd:startInd+numVars].reshape(r, -1)

                eigs, vecs = numpy.linalg.eigh(mat)

                Up = numpy.zeros((mat.shape))
                Un = numpy.zeros((mat.shape))

                for idx, i in enumerate(eigs):
                    Up[idx, idx] = max(0., i)
                    Un[idx, idx] = min(0., i)

                Up = reduce(numpy.dot, (vecs, Up, vecs.T))
                Un = reduce(numpy.dot, (vecs, Un, vecs.T))

                x = Up/mu
                z = Un*-1

                self.x[startInd:startInd+numVars] = x.flatten()
                self.z[startInd:startInd+numVars] = z.flatten()

                startInd += numVars

            return self.x, self.z

        def checkConv(self, it):

            if it >= self.maxIter:
                print('\n Maximum iteratations', self.maxIter, 'completed.\n')
                return True

            ePrimal = self.primalError()
            eDual = self.dualError()
            if (ePrimal <= self.tol and abs(self.primalEnergy() - self.dualEnergy()) <= self.eTol and eDual <= self.tol):
                print('\n Convergence Achieved. Finishing.\n')
                return True

            else:
                return False

        t0 = time.time()
        iterations = 0
        self.numVars = self.c.size
        self.numConst = self.b.size

        if self.x is None:

            x = []
            for i in self.blocks:
                mat = numpy.eye(i)
                x.extend(mat.flatten())

            self.x = numpy.asarray(x)

        if self.z is None:
            z = []
            for i in self.blocks:
                mat = numpy.eye(i)
                z.extend(mat.flatten())

            self.z = numpy.asarray(z)
        
        if self.A is None:
            print('\n No A matrix! Quitting!\n')
            return
        
        else:
            # compute AAt once
            t0 = time.time()
            self.AAt = numpy.dot(self.A, self.A.T)
            tf = time.time()
            print('Time for A dot AT ', round(tf-t0, 6), ' seconds')
            t0 = time.time()
            self.precond = numpy.linalg.pinv(self.AAt)
            #self.precond = rdm_tools.SVDInv(self.AAt)
            tf = time.time()
        
            print('Time for Computing Preconditioner ', round(tf-t0, 6), ' seconds')
        if self.b is None:
            print('\n No b matrix! Quitting!\n')
            return

        if self.c is None:
            print('\n c Matrix! Quitting!\n')
            return


        while not self.conv:
            t00 = time.time()
            self.y = yFromCG(self)
            t1 = time.time()
            self.tCG += t1 - t00

            if self.y is None:
                print('\n no y vector available, CG not converged.\n')
                break
        
            self.x, self.z = updateMx(self)
            
            t2 = time.time()
            self.tUM += t2 - t1

            # Calculate Primal and Dual errors
            ePrimal, eDual = self.primalError(), self.dualError()

            # Update $\mu$ every self.updateMu iterations
            if iterations % self.updateMu == 0:
                self.mu *= ePrimal/eDual

            iterations += 1
            # print('\n Iteration ', iterations)
            # print('Primal Energy = ', round(self.primalEnergy(), 9))
            # print('Dual Energy = ', round(self.dualEnergy(), 9))
            # print('Duality Gap = ', round(self.dualityGap(), 9))
            # print('Primal Error = ', round(self.primalError(), 9))
            # print('Dual Error = ', round(self.dualError(), 9), '\n')
            self.conv = checkConv(self, iterations)
            self.tTot = time.time()-t0

        # print('Timing')
        # print('Ave. CG time=', self.tCG/iterations)
        # print('Ave. UpdateM=', self.tUM/iterations)
        # print('Ave. BP =', self.tTot/iterations)
        # print('CG time =', self.tCG)
        # print('Update M =', self.tUM)
        # print('T tot =', self.tTot)
        return


def summarise_result(k2, d2ab, d2ab_fci, nuclear_rep, primal_energy, dual_energy, nround=None):
    if not nround is None:
        k2 = np.round(k2, nround)
        d2ab = np.round(d2ab, nround)
        d2ab_fci = np.round(d2ab_fci, nround)

    print('########## RESULTS ##########')
    print('K2:')
    print(k2)
    print('####################')
    print('D2ab:')
    print(d2ab)
    print('####################')
    print('TraceD:')
    print(np.trace(d2ab))
    print('####################')
    print('D2ab_FCI:')
    print(d2ab_fci)
    print('####################')
    print('norm(D2-D2FCI):')
    print(np.linalg.norm(d2ab - d2ab_fci, 'fro'))
    print('####################')
    print('NuclearRepulsion:')
    print(nuclear_rep)
    print('####################')
    print('PrimalEnergy:')
    print(primal_energy)
    print('####################')
    print('DualEnergy:')
    print(dual_energy)
    print('####################')
    print('DONE')


def run_example():
    print("=== START TEST: H2 Example 0.7414 ===")
    bases = ['STO-3G']  # , '6-31G']#, 'cc-pvdz']#, 'cc-pvtz']

    for bas in bases:
        t0 = time.time()

        testBP = boundaryPointSDP()
        print('Time to initialize BPSDP ', round(time.time() - t0, 5), ' seconds')

        geometry = [['H', (0.,0.,0.)], ['H', (0.,0.,0.7414)]]
        testBP.A, testBP.b, testBP.c, testBP.blocks, mats, nuclear_rep, k2, d2ab_fci = sdpTools.twoElectronAB(geometry, bas)

        testBP.maxIter = 100000
        testBP.solveSDP()

        cnt = 0
        solution = {}
        for i, block in enumerate(testBP.blocks):
            solution[mats[i]] = numpy.asarray(testBP.x[cnt:cnt + block ** 2]).reshape(block, -1)
            cnt += block ** 2

        d2ab = solution['D2ab']
        tF = time.time()

        print('Time for BPSDP test is ', str(round(tF - t0, 3)), ' seconds')
        summarise_result(k2, d2ab, d2ab_fci, nuclear_rep, testBP.primalEnergy(), testBP.dualEnergy(), nround=3)
        print("=== END TEST ===")


def run_sdp(molecule_geometry, bas):
    t0 = time.time()

    testBP = boundaryPointSDP()
    print('Time to initialize BPSDP ', round(time.time() - t0, 5), ' seconds')

    testBP.A, testBP.b, testBP.c, testBP.blocks, mats, nuclear_rep, k2, d2ab_fci = sdpTools.twoElectronAB(molecule_geometry, bas)

    testBP.maxIter = 100000
    testBP.solveSDP()

    cnt = 0
    solution = {}
    for i, block in enumerate(testBP.blocks):
        solution[mats[i]] = numpy.asarray(testBP.x[cnt:cnt + block ** 2]).reshape(block, -1)
        cnt += block ** 2

    d2ab = solution['D2ab']
    tF = time.time()

    print('Time for BPSDP test is ', str(round(tF - t0, 3)), ' seconds')
    summarise_result(k2, d2ab, d2ab_fci, nuclear_rep, testBP.primalEnergy(), testBP.dualEnergy(), nround=3)


def run_experiment(mol_name):
    step_size = 0.1
    steps = 2
    start = 1.0
    stop = start + step_size * steps
    xs = np.linspace(start, stop, steps)
    basis = 'STO-3G'

    for x in xs:
        mol = OPTIONS[mol_name]
        geometry = mol['geometry'](x)
        description = mol['description'](x)
        print("=== START TEST: %s ===" % description)
        run_sdp(geometry, basis)
        print("=== END TEST ===")


def main():
    # Check if argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Running example case.\n")
        run_example()

    else:
        # Get the argument (first argument after script name)
        user_arg = sys.argv[1].upper()

        # Validate the argument
        if user_arg not in OPTIONS.keys():
            print(f"Error: Invalid argument '{user_arg}'")
            print(f"Accepted arguments: {', '.join(OPTIONS.keys())}")
            sys.exit(1)

        print(f"Success: Received valid argument '{user_arg}'\n")
        run_experiment(user_arg)


if __name__ == "__main__":
    main()
