import itertools, numpy

def gen_trans_mat(gem_dim, bas_dim):
# Nick Rubin's code follows through 'return'

    bas = dict(zip(range(gem_dim), itertools.product(range(1,bas_dim+1), range(1,bas_dim+1))))

    bas_rev = dict(zip(bas.values(), bas.keys()))
    D2ab_abas = {}
    D2ab_abas_rev = {}
    cnt = 0
    for xx in range(bas_dim):
        for yy in range(xx+1, bas_dim):
            D2ab_abas[cnt] = (xx+1, yy+1)
            D2ab_abas_rev[(xx+1, yy+1)] = cnt
            cnt += 1
    D2ab_sbas = {}
    D2ab_sbas_rev = {}
    cnt = 0
    for xx in range(bas_dim):
        for yy in range(xx, bas_dim):
            D2ab_sbas[cnt] = (xx+1, yy+1)
            D2ab_sbas_rev[(xx+1, yy+1)] = cnt
            cnt += 1
    trans_mat = numpy.zeros((gem_dim, gem_dim))
    cnt = 0
    for xx in D2ab_abas.keys():
        i,j = D2ab_abas[xx]
        x1 = bas_rev[(i,j)]
        x2 = bas_rev[(j,i)]
        trans_mat[x1, cnt] = 1./numpy.sqrt(2)
        trans_mat[x2, cnt] = -1./numpy.sqrt(2)
        cnt += 1

    for xx in D2ab_sbas.keys():
        i,j = D2ab_sbas[xx]
        x1 = bas_rev[(i,j)]
        x2 = bas_rev[(j,i)]

        if x1 == x2:
            trans_mat[x1, cnt] = 1.0
        else:
            trans_mat[x1, cnt] = 1./numpy.sqrt(2)
            trans_mat[x2, cnt] = 1./numpy.sqrt(2)

        cnt += 1

    return trans_mat

def asym_mat(gem_dim, bas_dim):

    bas = dict(zip(range(gem_dim), itertools.product(range(1,bas_dim+1), range(1,bas_dim+1))))

    bas_rev = dict(zip(bas.values(), bas.keys()))
    cnt = 0
    for xx in bas.keys():
        i,j = D2ab_abas[xx]
        x1 = bas_rev[(i,j)]
        x2 = bas_rev[(j,i)]
        trans_mat[x1, cnt] = 1./numpy.sqrt(2)
        trans_mat[x2, cnt] = -1./numpy.sqrt(2)
        cnt += 1

    return trans_mat
