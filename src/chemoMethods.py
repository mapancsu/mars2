__author__ = 'Administrator'


from numpy import ix_
from scipy.linalg import norm
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from NetCDF import netcdf_reader


def efa(data):
    x = data['d']
    rz = x.shape
    n = 1
    ef = np.zeros((rz[0]-1, rz[0]))
    efl = np.zeros((rz[0]-1, rz[0]))
    while n <= rz[0]-1:
        u, svf, v = svd(x[0:n+1, :])
        l = svf**2
        nl = l.shape[0]
        ef[n-1, 0:nl] = l[0:nl]
        efl[n-1, 0:nl] = np.log10(l[0:nl])
        n += 1
    minvalue = efl.min().round()
    ns = efl.shape
    for i in range(0, ns[0]-1):
        for j in range(0, ns[1]-1):
            if efl[i, j] == 0:
                efl[i, j] = minvalue
    xforward = np.linspace(1, rz[0]-1)

    x = data['d'][::-1, :]
    n = 1
    eb = np.zeros((rz[0]-1, rz[0]))
    ebl = np.zeros((rz[0]-1, rz[0]))
    while n <= rz[0]-1:
        u, svb, v = np.linalg.svd(x[0:n+1, :])
        nl = svb.shape[0]
        l = svb**2
        eb[n-1, 0:nl] = l[0:nl]
        ebl[n-1, 0:nl] = np.log10(l[0:nl])
        n += 1
    minvalue = ebl.min().round()
    ns = ebl.shape
    for i in range(ns[0]-1):
        for j in range(ns[1]-1):
            if ebl[i, j] == 0:
                ebl[i, j] = minvalue
    xbackward = np.linspace(rz[0]-1, 1, -1)

    return{'efor': ef, 'efl': efl, 'ebac': eb, 'ebl': ebl}

def pure(d, nr, f):
    rc = d.shape
    p = np.zeros((nr, rc[1]))
    s = np.zeros((nr, rc[1]))
    imp = list()
    s[0, :] = np.std(d, axis=0, ddof=1)
    m = np.mean(d, axis=0)
    ll = s[0, :]**2 + m**2
    f = f/100.00*m.max()
    pp = s[0, :]/(m+f)
    imp.append(np.argmax(pp))
    l = np.power(s[0, :]**2+(m+f)**2, 0.5)
    dl = np.zeros(d.shape)
    for j in range(0, rc[1]):
        dl[:, j] = d[:, j]/l[j]
    c = np.dot(dl.T, dl)/rc[0]
    w = np.zeros((nr, rc[1]))
    w[0, :] = ll / (l**2)
    p[0, :] = w[0, :]*pp
    s[0, :] = w[0, :]*s[0, :]

    for i in range(1, nr):
        for j in range(0, rc[1]):
            dm = wmat(c, imp[0:i], i, j)
            w[i, j] = np.linalg.det(dm)
            p[i, j] = p[0, j]*w[i, j]
            s[i, j] = s[0, j]*w[i, j]
        imp.append(np.argmax(p[i, :]))
    sn = d[:, imp]
    sp = sn.T
    for bit in range(0, sp.shape[0]):
        sr = np.linalg.norm(sp[bit, :])
        sp[bit, :] = sp[bit, :]/sr
    return{'SP': sp, 'IMP': imp}

def wmat(c, imp, irank, jvar):
    dm = np.zeros((irank+1, irank+1))
    dm[0, 0] = c[jvar, jvar]
    for k in range(1, irank+1):
        kvar = imp[k-1]
        dm[0, k] = c[jvar, kvar]
        dm[k, 0] = c[kvar, jvar]
        for kk in range(1, irank+1):
            kkvar = imp[kk-1]
            dm[k, kk] = c[kvar, kkvar]
    return dm

def mcr_als(d, dn, sst, sstn, sigma2, crs, options):
    if d.shape[0] == crs.shape[0]:
        conc = crs
        spec = np.dot(np.dot(np.linalg.inv(np.dot(conc.T, conc)), conc.T), d)
    elif d.shape[1] == crs.shape[1]:
        spec = crs
        conc = np.dot(np.dot(d, spec.T), np.linalg.inv(np.dot(spec, spec.T)))
    else:
        print('please import right initial estimation')

    if options['chrom'][0] == "fnnls":
        conc2 = np.zeros(conc.shape)
        for j in range(0, conc.shape[0]):
            a = fnnls(np.dot(spec, spec.T), np.dot(spec, d[j, :].T), tole='None')
            conc2[j, :] = a['xx']
        conc = conc2

    if options['chrom'][1] == "average":
        mod = [1.1, 2]
        conc2 = unimod(conc, mod[0], mod[1])
        conc = conc2

    if options['mass'][0] == "fnnls":
        spec2 = np.zeros(spec.shape)
        for j in range(0, spec.shape[1]):
            a = fnnls(np.dot(conc.T, conc), np.dot(conc.T, d[:, j]), tole='None')
            spec2[:, j] = a['xx']
    spec = spec2

    res = d-np.dot(conc, spec)
    resn = dn-np.dot(conc, spec)
    u = np.sum((np.power(res, 2)))
    un = np.sum((np.power(resn, 2)))
    sigma = np.power(u / (d.shape[0]*d.shape[1]), 0.5)
    # print(sigma2)
    # print(sigma)
    change = (sigma2 - sigma)/sigma
    lof_pca = np.power((u/sst), 0.5)*100
    lof_exp = np.power((un/sstn), 0.5)*100
    r2 = (sstn-un)/sstn
    return conc, spec, res, resn, u, un, sigma, change, lof_pca, lof_exp, r2



def MCRALS(x, pc, pures, options):
        tolsigma = 0.1
        niter = 0
        idev = 0
        rttic = x['rt']
        dn = x['d']
        u, s, v, d, sd = pcarep(dn, pc)
        sstn = np.sum(np.power(dn, 2))
        sst = np.sum(np.power(d, 2))
        nit = 20
        while niter < nit:
            niter += 1
            conc, spec, res, resn, u, un, sigma, change, lof_pca, lof_exp, r2 = \
                mcr_als(d, pc, pures, options)
            if change < 0.0:
                idev += 1
            else:
                idev = 0
            change = np.dot(100, change)
            lof_pca = np.power((u/sst), 0.5)*100
            lof_exp = np.power((un/sstn), 0.5)*100
            r2 = (sstn-un)/sstn
            if change > 0 or niter == 0:
                print(change)
                copt = conc
                sopt = spec
                itopt = niter+1
                sdopt = np.array([lof_pca, lof_exp])
                ropt = res
                r2opt = r2
            if abs(change) < tolsigma:
                print('CONVERGENCE IS ACHIEVED, STOP!!!')
                break
            if idev >= 20:
                print('FIT NOT IMPROVING FOR 20 TMES CONSECUTIVELY (DIVERGENCE?), STOP!!!')
                break
            print(str(niter))
        result = {'copt': copt, 'sopt': sopt, 'sdopt': sdopt, 'r2opt': r2opt, 'ropt': ropt, 'itopt': itopt}
        RESU = {}
        RESU['pc'] = pc
        RESU['tic'] = np.sum(dn, axis=1)
        RESU['rttic'] = rttic
        RESU['rt'] = rttic[np.sort(np.argmax(copt, 0))]
        sums = np.zeros(np.shape(np.transpose(copt)))
        mass_norm = np.zeros(sopt.shape)
        index = np.argsort(np.argmax(copt, axis=0))
        for ind, val in enumerate(index):
            cc = result['copt'][:, val:val+1]
            ss = result['sopt'][val:val+1, :]
            sums[ind, :] = np.array(np.sum(np.dot(cc, ss), 1), ndmin=2)
            mass_norm[ind, :] = ss/norm(ss)
        RESU['chro'] = sums
        RESU['spec'] = mass_norm
        RESU['r2'] = result['r2opt']
        return RESU

def pcarep(xi, nf):
    u, s, v = np.linalg.svd(xi)
    x = np.dot(np.dot(u[:, 0:nf], np.diag(s[0:nf])), v[0:nf, :])
    res = xi - x
    sst1 = np.power(res, 2).sum()
    sst2 = np.power(xi, 2).sum()
    sigma = sst1/sst2*100
    return u, s, v, x, sigma

def fnnls(xtx, xty, tole):
    if tole == 'None':
        tol = 10*np.spacing(1)*norm(xtx, 1)*max(xtx.shape)
    mn = xtx.shape
    P = np.zeros(mn[1])
    Z = np.array(range(1, mn[1]+1), dtype='int64')
    xx = np.zeros(mn[1])
    ZZ = Z-1
    w = xty-np.dot(xtx, xx)
    iter = 0
    itmax = 30*mn[1]
    z = np.zeros(mn[1])
    while np.any(Z) and np.any(w[ZZ] > tol):
        t = ZZ[np.argmax(w[ZZ])]
        P[t] = t+1
        Z[t] = 0
        PP = np.nonzero(P)[0]
        ZZ = np.nonzero(Z)[0]
        nzz = np.shape(ZZ)
        if len(PP) == 1:
            z[PP] = xty[PP]/xtx[PP, PP]
        elif len(PP) > 1:
            z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[ix_(PP, PP)]))
        z[ZZ] = np.zeros(nzz)
        while np.any(z[PP] <= tol) and iter < itmax:
            iter += 1
            qq = np.nonzero((tuple(z <= tol) and tuple(P != 0)))
            alpha = np.min(xx[qq] / (xx[qq] - z[qq] + np.spacing(1)))
            xx = xx + alpha*(z - xx)
            ij = np.nonzero(tuple(np.abs(xx) < tol) and tuple(P != 0))
            Z[ij[0]] = ij[0]+1
            P[ij[0]] = np.zeros(max(np.shape(ij[0])))
            PP = np.nonzero(P)[0]
            ZZ = np.nonzero(Z)[0]
            nzz = np.shape(ZZ)
            if len(PP) == 1:
                z[PP] = xty[PP]/xtx[PP, PP]
            elif len(PP) > 1:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[ix_(PP, PP)]))
            z[ZZ] = np.zeros(nzz)
        xx = np.copy(z)
        w = xty - np.dot(xtx, xx)
    return{'xx': xx, 'w': w}

def unimod(c, rmod, cmod):
    ns = c.shape[1]
    imax = np.argmax(c, axis=0)
    for j in range(0, ns):
        rmax = c[imax[j], j]
        k = imax[j]
        while k > 0:
            k = k-1
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 1e-30
                    if cmod == 1:
                        c[k, j] = c[k+1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k+1, j])/2
                            c[k+1, j] = c[k, j]
                            k = k+2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
        rmax = c[imax[j], j]
        k = imax[j]

        while k < c.shape[0]-1:
            k = k+1
            if k==53:
                k=53
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 1e-30
                    if cmod == 1:
                        c[k, j] = c[k-1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k-1, j])/2
                            c[k-1, j] = c[k, j]
                            k = k-2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
    return c

def backremv(x, seg):
    mn = np.shape(x)
    bak2 = np.zeros(mn)
    for i in range(0, mn[1]):
        tiab = []
        reg = []
        for j in range(0, len(seg)):
            tt = range(seg[j][0],seg[j][1])
            tiab.extend(x[tt, i])
            reg.extend(np.arange(seg[j][0], seg[j][1]))
        rm = reg - np.mean(reg)
        tm = tiab - np.mean(tiab)
        b = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
        s = np.mean(tiab)-np.dot(np.mean(reg), b)
        b_est = s+b*np.arange(mn[0])
        bak2[:, i] = x[:, i]-b_est
    bak = x-bak2
    return bak2, bak

if __name__ == '__main__':

    filename = 'F:\MARS\data_save\D6/73.CDF'
    ncr = netcdf_reader(filename, True)
    options = {'chrom':['fnnls', 'average'], 'mass':['fnnls']}
    x = ncr.mat(1850, 1890, 1)
    # ax = plt.figure()
    # ax.add_subplot(111)
    # ax.show()
    plt.plot(x['d'])
    plt.show()

    ### test MCR_ALS
    pures = pure(x['d'].T, 3, 10)
    # RESU = MCRALS(x, 3, pures['SP'], options)

    # moptions = {'pw':80, 'w':3, 'thres':0.8}
    # MSRT = {'ms': RESU['spec'], 'rt': RESU['rt']}

    # plt.plot(RESU['chro'].T)
    # plt.show()

    # data = {'MSRT': MSRT, 'options': moptions,
    #         'ncr': ncr}

    # data = {'MSRT': MSRT,'options': moptions, 'fn':filename}
    #
    # output1 = open('data5.pkl', 'wb')
    # pickle.dump(data, output1)
    # output1.close()
    #
    # pkl_file = open('data5.pkl')
    # data1 = pickle.load(pkl_file)
    #
    # save_path = 'random_walk.png'

### test fnnls
    # tt = np.zeros(s.shape[1])
    # for j in range(0, s.shape[1]):
    #     t1 = time.time()
    #     a = fnnls(np.dot(c.T, c), np.dot(c.T, d[:, j]), tole='None')
    #     s[:, j] = a['xx']
    #     t2 = time.time()
    #     tt[j] = t2-t1
    # tsum = np.sum(tt)

