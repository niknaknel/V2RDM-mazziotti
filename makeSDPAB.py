import numpy, time, itertools
numpy.set_printoptions(linewidth=10000)
from collections import defaultdict

'''

Generate A from user-provided matrix variables, and constraints

'''
# TODO: 2G

class makeSDPConstraints(object):
    def __init__(self, arg):

        self.canonical_order = ['D2aa', 'D2ab', 'D2bb', 'D1a', 'D1b', 'Q2aa', 'Q2ab', 'Q2bb', 'Q1a', 'Q1b', 'G2ab', 'G2ba', 'G2big']
        self.positivity = arg[0]
        self.r = arg[1]
        self.Na = arg[2][0]
        self.Nb = arg[2][1]
        self.N = self.Na + self.Nb
        
        self.parsePositivity()


        if self.varlist == None:
            self.varlist = ['D2aa', 'D2ab', 'D2bb', 'D1a', 'D1b', 'Q2aa', 'Q2ab', 'Q2bb', 'Q1a', 'Q1b']

        self.matParams = self.genSysInfo()
        self.startInds = self.findStartInds()
        nv, nc = 0, 0
        print('Printing System Info')
        print('Matrix Rank NumVar numConst')
        for mat in self.varlist:
                print(mat, self.matParams[mat])
                nv += self.matParams[mat][1]
                nc += self.matParams[mat][2]
        print('\n Total Variables = ', nv)
        print(' Total Constraints = ', nc)

    def parsePositivity(self):

        pos = list(self.positivity)
        self.varlist = []

        for p in pos:
                print(p)
                if p == 'D':
                        self.varlist.extend(['D2aa', 'D2ab', 'D2bb', 'D1a', 'D1b', 'Q1a', 'Q1b'])

                if p == 'Q':
                        self.varlist.extend(['Q2aa', 'Q2ab', 'Q2bb'])

                if p == 'G':
                        self.varlist.extend(['G2big', 'G2ab', 'G2ba'])
        
        return
    
    def findStartInds(self):
        
        startInd = {}
        cnt = 0
        for mat in self.canonical_order:
                if mat in self.varlist:
                    startInd[mat] = cnt
                    cnt += self.matParams[mat][1]
        
        return startInd

    def genSysInfo(self):

        r = self.r
        self.Ca, lc = contractDict(r)
        rs = r**2
        rA = r*(r-1)//2
        rGbig = 2*(r**2)
        N = self.N
        # dict: MatrixName, rank, NumVar, numConstr (trace + herm... so far)
        ## Getting rid of hermiticy constraints and constraints on d1a/b
        matParams = {'D1a' :(r,  r**2, 0), # Trace, Herm
                     'D1b' :(r,  r**2, r**2), # D1b==D1a
                     'Q1a' :(r,  r**2,  r**2), # Herm, DQI
                     'Q1b' :(r,  r**2,  r**2),
                     'D2aa':(rA, rA**2, 1 + lc), # Trace, Herm, contract
                     'D2bb':(rA, rA**2, 1 + lc + rA**2), # AA == BB
                     'D2ab':(rs, rs**2, 1 + 2*r**2 + 1), # Trace, Herm, 2*contract, spin
                     'Q2aa':(rA, rA**2, rA**2), # linear map and herm
                     'Q2ab':(rs, rs**2, rs**2), # linear map and herm
                     'Q2bb':(rA, rA**2, rA**2), # linear map and herm
                     'G2big':(rGbig, rGbig**2, 4*rs**2),
                     'G2ab' :(rs, rs**2, rs**2 + 1),
                     'G2ba' :(rs, rs**2, rs**2 + 1)}

        return matParams

    def makeD2aa(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['D2aa'][0]

        # Trace D2aa = Ns*(Ns-1)/2
        Na = self.Na
        tr = Na*(Na-1)/2

        A, b, cnt = addTraceM(A, b, r, tr, cnt, startInd)
        
        startD1 = self.startInds['D1a']
        d1Rank = self.matParams['D1a'][0]
        Ca = self.Ca

        for ij in Ca.keys():
                for func in Ca[ij]:
                        i,k,j,k = func
                        
                        ik, fik = ASymBas(i,k,d1Rank)
                        jk, fjk = ASymBas(j,k,d1Rank)

                        d2Ind = ik*r + jk + startInd
                        A[cnt, d2Ind] = fik*fjk
                
                d1Ind = i*d1Rank + j + startD1
                A[cnt, d1Ind] = -1.*(Na-1.)
                cnt += 1

        return A, b

    def makeD2bb(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['D2bb'][0]

        # Trace D2bb = Nb*(Nb-1)/2
        Nb = self.Nb
        tr = Nb*(Nb-1)/2

        A, b, cnt = addTraceM(A, b, r, tr, cnt, startInd)
        
        Ca = self.Ca
        startD1b = self.startInds['D1b']
        d1Rank = self.matParams['D1b'][0]
        for ij in Ca.keys():
                for func in Ca[ij]:
                        i,k,j,k = func
                        
                        ik, fik = ASymBas(i,k,d1Rank)
                        jk, fjk = ASymBas(j,k,d1Rank)

                        d2Ind = ik*r + jk + startInd
                        A[cnt, d2Ind] = fik*fjk
                
                d1Ind = i*d1Rank + j + startD1b
                A[cnt, d1Ind] = -1.*(Nb-1)
                cnt += 1
        
        # D2bb = D2aa for singlet

        d2aaStart = self.startInds['D2aa']
        for ij in range(r):
                for kl in range(r):
                        aaInd = d2aaStart + r*ij + kl
                        bbInd = startInd + r*ij + kl
                        A[cnt, aaInd] = 1.
                        A[cnt, bbInd] = -1.

                        cnt += 1

        return A, b
    
    def makeD2ab(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['D2ab'][0]
        # Trace D2ab = Na*Nb
        Na, Nb = self.Na, self.Nb
        tr = Na*Nb
        A, b, cnt = addTraceM(A, b, r, tr, cnt, startInd)
        
        startD1 = self.startInds['D1a']
        d1Rank = self.matParams['D1a'][0]
        for i in range(d1Rank):
            for j in range(d1Rank):
                for k in range(d1Rank):
                    d2Ind = d1Rank**2*(d1Rank*i + k) + d1Rank*j + k
                    A[cnt, startInd+d2Ind] = 1.

                d1Ind = startD1 + d1Rank*i + j
                A[cnt, d1Ind] = -1.*Nb
                cnt += 1
       
       # Contract D2ab: Na*D1bij = sum_k 2Dabikjk
        #startD1 =  self.matParams['D2ab'][1] + self.matParams['D2aa'][1]
        startD1 = self.startInds['D1b']
        d1Rank = self.matParams['D1b'][0]
        for i in range(d1Rank):
            for j in range(d1Rank):
                for k in range(d1Rank):
                    d2Ind = d1Rank**2*(d1Rank*k + i) + d1Rank*k + j
                    A[cnt, startInd+d2Ind] = 1.

                d1Ind = startD1 + d1Rank*i + j
                A[cnt, d1Ind] = -1.*Na
                cnt += 1
        
        # Add spin Ns = sum_ij D2ab ijji
        Ms = Na-Nb
        S = 0.5*(Na-Nb)

        spinCons = 0.5*(Na+Nb) + Ms**2 - S*(S+1)

        #Ms = (Na-Nb)/2
        #spinCons = Ms*(Ms+1)

        for i in range(d1Rank):
            for j in range(d1Rank):
                d2Ind = startInd + d1Rank**2*(d1Rank*i + j) + d1Rank*j + i
                A[cnt, d2Ind] = 1.

        b[cnt] = spinCons
        cnt += 1

        return A, b

    def makeQ2aa(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['Q2aa'][0]


        startD1 = self.startInds['D1a']
        d1Rank = self.matParams['D1a'][0]
        startD2aa = self.startInds['D2aa']

        for i in range(d1Rank):
            for j in range(i+1, d1Rank):
                for k in range(d1Rank):
                    for l in range(k+1, d1Rank):

                        ik = 1. if i == k else 0.
                        jl = 1. if j == l else 0.

                        indQ2 = r*aSymBas(i,j,d1Rank) + aSymBas(k,l,d1Rank)
                        indD2 = r*aSymBas(k,l,d1Rank) + aSymBas(i,j,d1Rank)

                        b[cnt] = ik*jl*-1.

                        A[cnt, indQ2 + startInd] = -1.
                        A[cnt, indD2 + startD2aa] = 1.

                        el = startD1 + i*d1Rank + k
                        A[cnt, el] += jl*-1.

                        el = startD1 + j*d1Rank + l
                        A[cnt, el] += ik*-1.

                        cnt += 1

        return A, b
    
    def makeQ2bb(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['Q2bb'][0]

        startD1 = self.startInds['D1b']
        d1Rank = self.matParams['D1b'][0]
        startD2aa = self.startInds['D2bb']

        for i in range(d1Rank):
            for j in range(i+1,d1Rank):
                for k in range(d1Rank):
                    for l in range(k+1,d1Rank):
                        ik = 1. if i == k else 0.
                        jl = 1. if j == l else 0.

                        ij = i*d1Rank+j
                        kl = k*d1Rank + l

                        indQ2 = r*aSymBas(i,j,d1Rank) + aSymBas(k,l,d1Rank)
                        indD2 = r*aSymBas(k,l,d1Rank) + aSymBas(i,j,d1Rank)

                        b[cnt] = ik*jl*-1.

                        A[cnt, indQ2 + startInd] = -1.
                        A[cnt, indD2 + startD2aa] = 1.

                        el = startD1 + i*d1Rank + k
                        A[cnt, el] += jl*-1.

                        el = startD1 + j*d1Rank + l
                        A[cnt, el] += ik*-1.

                        cnt += 1
        
        return A, b

    def makeQ2ab(self):
        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['Q2ab'][0]
        startD1 = self.startInds['D1a']
        d1Rank = self.matParams['D1a'][0]
        startD2ab = self.startInds['D2ab']

        # Linear Map
        for ij in range(r):
                for kl in range(r):
                        i,j = SymBas(ij, d1Rank)
                        k,l = SymBas(kl, d1Rank)
                        ik = 1. if i == k else 0.
                        jl = 1. if j == l else 0.
                        
                        indQ2 = r*(d1Rank*i + j) + d1Rank*k + l
                        indD2 = r*(d1Rank*k + l) + d1Rank*i + j

                        b[cnt] = ik*jl*-1.

                        A[cnt, indQ2 + startInd] = -1. 
                        A[cnt, indD2 + startD2ab] = 1.

                        el = startD1 + i*d1Rank + k
                        A[cnt, el] += jl*-1.

                        el = startD1 + j*d1Rank + l
                        A[cnt, el] += ik*-1.

                        cnt += 1

        return A, b

    def makeG2big(self):
        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        rBig = self.matParams['G2big'][0] # rank of big block is 2*r**2
        rMat = rBig//2 # rank of any given G block
        d1Rank = self.matParams['D1a'][0]
    
        # AABB and BBAA map
        # AABB ijkl = -D2iljk + il* D1aik
        # BBAA ijkl = -D2likj + il* D1bik
        # BBBB ijkl = ilD1bik - bbD2ilkj
        # AAAA ijkl = ilD1aik - aaD2ilkj
    
        startD2ab = self.startInds['D2ab']
        startD1a = self.startInds['D1a']
        startD1b = self.startInds['D1b']
    
        asymRank = self.matParams['D2aa'][0]
        startD2aa = self.startInds['D2aa']
        startD2bb = self.startInds['D2bb']
    
        for i in range(d1Rank):
                for j in range(d1Rank):
                        for k in range(d1Rank):
                                for l in range(d1Rank):
                                        
                                        jl = 1. if j == l else 0.
                                       
                                        aabbD2Ind = startD2ab + d1Rank**2*(d1Rank*i + l) + d1Rank*j + k
                                        bbaaD2Ind = startD2ab + d1Rank**2*(d1Rank*l + i) + d1Rank*k + j
                                        d1aInd = d1Rank*i + k + startD1a
                                        d1bInd = d1Rank*i + k + startD1b
                                        
                                        indAABB = startInd + rBig*(d1Rank*i + j) + d1Rank*k + l + rMat 
                                        indBBAA = startInd + rBig*(d1Rank*i + j) + d1Rank*k + l + rMat*rBig
   
                                        A[cnt, indAABB] = -1.
                                        A[cnt, aabbD2Ind] = 1.
    
                                        cnt += 1
                                        
                                        A[cnt, indBBAA] = -1.
                                        A[cnt, bbaaD2Ind] = 1.
    
                                        cnt += 1
    
                                        # Do AAAA and BBBB
                                        indAAAA = startInd + rBig*(d1Rank*i + j) + d1Rank*k + l
                                        indBBBB = startInd + rBig*(d1Rank*i + j) + d1Rank*k + l + rMat + rMat*rBig
    
                                        if (i == l or j == k):
                                                A[cnt, indAAAA] = -1.
                                                A[cnt, d1aInd] = jl
                                                cnt += 1
    
                                                A[cnt, indBBBB] = -1.
                                                A[cnt, d1bInd] = jl
                                                cnt += 1
                                                
                                                continue
    
                                        else:
                                                
                                                d2Ind = 0
                                                fac = 1.
    
                                                if i > l:
                                                        d2Ind += aSymBas(l, i, d1Rank)*asymRank
                                                        fac *= -1.
                                                else:
                                                        d2Ind += aSymBas(i, l, d1Rank)*asymRank
    
                                                if k > j:
                                                        d2Ind += aSymBas(j, k, d1Rank)
                                                        fac *= -1.
                                                else:
                                                        d2Ind += aSymBas(k, j, d1Rank)
    
                                                A[cnt, indAAAA] = -1.
                                                A[cnt, d1aInd] = jl
                                                A[cnt, d2Ind + startD2aa] = fac*-1.
                                                cnt += 1
    
                                                A[cnt, indBBBB] = -1.
                                                A[cnt, d1bInd] = jl
                                                A[cnt, d2Ind + startD2bb] = fac*-1
                                                cnt += 1
        return A, b
        
    def makeG2ab(self):
        
        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['G2ab'][0]
        d1Rank = self.matParams['D1a'][0]

        startD1a = self.startInds['D1a']
        startD2ab = self.startInds['D2ab']
        
        # Map
        # Gab ijkl = jl * D1a[ik] - Dab[ilkj]

        for i in range(d1Rank):
                for j in range(d1Rank):
                        for k in range(d1Rank):
                                for l in range(d1Rank):
                                        jl = 1. if j == l else 0.

                                        d1Ind = d1Rank*i + k + startD1a
                                        d2Ind = d1Rank**2*(d1Rank*i + l) + d1Rank*k + j + startD2ab
                                        g2Ind = d1Rank**2*(d1Rank*i + j) + d1Rank*k + l + startInd

                                        A[cnt, d1Ind] = jl
                                        A[cnt, d2Ind] = -1.
                                        A[cnt, g2Ind] = -1.

                                        cnt += 1
    

        # Maximal spin constraint?

        for kl in range(r):
                for i in range(d1Rank):

                        g2Ind = kl*r + i*d1Rank + i + startInd
                        A[cnt, g2Ind] = 1.

        cnt += 1

        return A, b

    def makeG2ba(self):
        
        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['G2ba'][0]
        d1Rank = self.matParams['D1b'][0]

        startD1b = self.startInds['D1b']
        startD2ab = self.startInds['D2ab']
        
        # Map
        # Gba ijkl = jl * D1b[ik] - Dab[lijk]

        for i in range(d1Rank):
                for j in range(d1Rank):
                        for k in range(d1Rank):
                                for l in range(d1Rank):
                                        jl = 1. if j == l else 0.

                                        d1Ind = d1Rank*i + k + startD1b
                                        d2Ind = d1Rank**2*(d1Rank*l + i) + d1Rank*j + k + startD2ab
                                        g2Ind = d1Rank**2*(d1Rank*i + j) + d1Rank*k + l + startInd

                                        A[cnt, d1Ind] = jl
                                        A[cnt, d2Ind] = -1.
                                        A[cnt, g2Ind] = -1.

                                        cnt += 1
        # Maximal spin constraint?

        for kl in range(r):
                for i in range(d1Rank):

                        g2Ind = kl*r + i*d1Rank + i + startInd
                        A[cnt, g2Ind] = 1.

        cnt += 1
                                        
        return A, b


    def makeD1a(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['D1a'][0] # rank of D1

        return A, b
    
    def makeD1b(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['D1b'][0] # rank of D1
        startD1a = self.startInds['D1a']
        # Trace
        tr = self.Nb

        # D1a = D1b
        for i in range(r):
                for j in range(i,r):
                        indb = r*i + j + startInd
                        indA = r*i + j + startD1a

                        A[cnt, indb] = 1.
                        A[cnt, indA] = -1.
                        cnt += 1


        return A, b

    def makeQ1a(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['Q1a'][0] # rank of Q1

        # Add D + Q = I
        d1Start = self.startInds['D1a']
        for i in range(r):
                for j in range(r):
                        d1Ind = r*i + j + d1Start
                        q1Ind = r*j + i + startInd

                        if i == j:
                                num = 1.
                        else:
                                num = 0.

                        A[cnt, d1Ind] = 1.
                        A[cnt, q1Ind] = 1.

                        b[cnt] = num
                        cnt += 1
        
        return A, b

    def makeQ1b(self):

        A = self.A
        b = self.b
        cnt = self.runningCnt
        startInd = self.startVar
        r = self.matParams['Q1b'][0] # rank of Q1

        # Add D + Q = I
        d1Start = self.startInds['D1b']
        for i in range(r):
                for j in range(r):
                        d1Ind = r*i + j + d1Start
                        q1Ind = r*j + i + startInd

                        if i == j:
                                num = 1.
                        else:
                                num = 0.

                        A[cnt, d1Ind] = 1.
                        A[cnt, q1Ind] = 1.

                        b[cnt] = num
                        cnt += 1
        
        return A, b
    
    def makeAB(self):
        '''
        This function will generate and return A (and b)

        '''
        t0 = time.time()
        self.numVars = 0
        self.numCons = 0
        self.runningCnt = 0
        self.startVar = 0
        self.blocks = []

        for mat in self.varlist:
            self.numVars += self.matParams[mat][1] # number of vars in matrix mat
            self.numCons += self.matParams[mat][2] # ''        cons ''
        numVars, numCons = self.numVars, self.numCons
        self.A = numpy.zeros((numCons, numVars))
        self.b = numpy.zeros((numCons))

        for mat in self.canonical_order:
            if mat not in self.varlist:
                continue
            
            if mat is 'G2ab':
                
                self.A, self.b = self.makeG2ab()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'G2ba':
                
                self.A, self.b = self.makeG2ba()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'G2big':
                
                self.A, self.b = self.makeG2big()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'Q2aa':

                self.A, self.b = self.makeQ2aa()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]
            
            if mat is 'Q2bb':

                self.A, self.b = self.makeQ2aa()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'Q2ab':

                self.A, self.b = self.makeQ2ab()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'D2aa':

                self.A, self.b = self.makeD2aa()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]
            
            if mat is 'D2bb':

                self.A, self.b = self.makeD2bb()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'D2ab':

                self.A, self.b = self.makeD2ab()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            if mat is 'Q1a':

                self.A, self.b = self.makeQ1a()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]
            
            if mat is 'Q1b':

                self.A, self.b = self.makeQ1b()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]
            
            if mat is 'D1b':

                self.A, self.b = self.makeD1b()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

            elif mat is 'D1a':

                self.A, self.b = self.makeD1a()

                self.blocks.append(self.matParams[mat][0])
                self.startVar += self.matParams[mat][1]
                self.runningCnt += self.matParams[mat][2]

        tF = time.time()

        print('Time to make SDP is ', str(tF-t0), ' seconds')
        return self.A, self.b, self.blocks

def addTraceM(A_mat, b_mat, rank, s, cntC, cntV):
    '''
    Add constraint to matrix A_mat s.t.
    trace=s for variable matrix that starts at cntV and rank=rank

    return new A_mat, b_mat, and new counter
    '''

    cnt = cntC
    startInd = cntV
    r = rank

    for i in range(r):
        ind = r*i + i + startInd
        A_mat[cnt, ind] = 1.

    b_mat[cnt] = s
    cnt += 1

    return A_mat, b_mat, cnt

def aSymBas(i,j,r):

        ind = 0
        if i != 0:
                for alpha in range(0, i):
                        ind += r - alpha - 1
        
        ind += j - i - 1

        return ind

def SymBas(ij,r):

        symbas = list(itertools.product(range(r), repeat=2))

        return symbas[ij][:]

def ASymBas(i,j,r):

        fac = 1.
        bas = list(itertools.combinations(range(r), 2))
        if i > j:
                fac *= -1.
                ind = bas.index((j,i))

        else:
                ind = bas.index((i,j))

        return ind, fac
        


def contractDict(r):

        bas = list(itertools.product(range(r), repeat=4))

        d = defaultdict(list)

        for func in bas:
                i,j,k,l = func
                
                if j != l:continue
                elif i == j: continue
                elif k == l: continue
                else:
                        d[i,k].append(func)

        return d, len(d.keys())


if __name__ == '__main__':

    mySDP = makeSDPConstraints((None, 2, 1))
    print(mySDP.varlist)
    A, b = mySDP.makeAB()
    print('Printing A for r=2, N=2')
    print(A)
    print()
    print('Print b for r=2, N=2')
    print(mySDP.b)
