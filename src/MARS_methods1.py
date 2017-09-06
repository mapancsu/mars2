from __future__ import division
__author__ = 'Administrator'
'''
MARS (Massspectrum-Assited Resolution of Signal)

Created on 2015-11-10

@author: pandemar

'''

from numpy import zeros, hstack, ix_
from matplotlib.ticker import FormatStrFormatter
from scipy.linalg import norm
from numpy.linalg import svd
import scipy.io as nc
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import pickle
import scipy

from NetCDF import netcdf_reader


def pure(d, nr, f):
    rc = d.shape
    p = np.zeros((nr, rc[1]))
    s = np.zeros((nr, rc[1]))
    imp = list()
    s[0, :] = np.std(d, axis=0, ddof=1)
    m = np.mean(d, axis=0)
    ll = s[0, :]**2 + m**2
    f = f/100*m.max()
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

def fnnls(x, y, tole):
    xtx = np.dot(x, x.T)
    xty = np.dot(x, y.T)
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
            alpha = np.min(xx[qq] / (xx[qq] - z[qq]))
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

def unimod(c, rmod, cmod, imax=None):
    ns = c.shape[1]
    if imax == None:
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
                        c[k, j] = 0 #1e-30
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

def pcarep(xi, nf):
    u, s, v = np.linalg.svd(xi)
    x = np.dot(np.dot(u[:, 0:nf], np.diag(s[0:nf])), v[0:nf, :])
    res = xi - x
    sst1 = np.power(res, 2).sum()
    sst2 = np.power(xi, 2).sum()
    sigma = np.power(sst1/sst2, 0.5)*100
    return u, s, v, x, sigma

def findTZ(ncr, rts, mss, pw, w, thre):
    pos = 0
    max_pc = 5
    tic_dict = ncr.tic()
    tic_rt = tic_dict['rt']
    sc = np.searchsorted(tic_rt, rts)
    sta = max([0, sc-pw])
    end = min([sc+pw, len(tic_rt)-1])
    seg = [sta, end+1]
    segx = ncr.mat(seg[0], seg[1]-1, 1)
    resp, eig, ir = MSWFA(segx['d'], mss, max_pc, w, thre)
    vseg = np.arange(seg[0], seg[1])
    if resp[0, 1] != 0:
        tz = vseg[resp][0, :]
        x = segx['d'][resp[0, 0]:resp[0, 1], :]
        reg = range(resp[0, 0], resp[0, 1])
        irr = [ir[0, 0]-resp[0, 0]+1, ir[0, 1]-resp[0, 0]+2]
        pos = np.argmax(eig[reg])
    else:
        tz = np.zeros(2)
        x = np.array([], ndmin=2)
        irr = [0, 0]
    return tz, seg, x, segx['d'], pos+1, irr

def MSWFA(d, ms, max_pc, w, thres):
    s = np.zeros(d.shape[0])
    for j in range(0, d.shape[0]-w):
        d1 = d[j:j+w, :]
        u, ss, e = np.linalg.svd(d1)
        M = np.dot(np.dot(np.dot(e[0:max_pc, :], ms.T), ms), e[0:max_pc, :].T)
        S = np.diag(M)
        if np.max(s) == 0:
            s[0:j+math.ceil((w-1)/2)] = S[0]
        if j+w == d.shape[0]-1:
            s[j + math.ceil((w - 1) / 2):d.shape[0]] = S[0]
        s[j+math.ceil((w-1)/2)] = S[0]
    maxs = np.max(s)
    pos = np.argmax(s)

    # plt.plot(s)
    # plt.show()

    region = np.array([0, len(s)-1], ndmin=2)
    ir = np.array([0, len(s)-1], ndmin=2)
    for i, val in enumerate(range(pos, 0, -1)):
        if s[val-1] <= 0.95*maxs and ir[0, 0] == 0:
            ir[0, 0] = val
        if (s[val-1] >= s[val] and s[val-1] <= 0.95*maxs) or s[val-1] <= 0.1*maxs:
            region[0, 0] = val
            break
    for i, val in enumerate(range(pos, len(s)-1)):
        if s[val+1] <= 0.95*maxs and ir[0, 1] == 100:
            ir[0, 1] = val+1
        if (s[val+1] >=s [val] and s[val+1] <= 0.95*maxs) or s[val+1] <= 0.1*maxs:
            region[0, 1] = val+1
            break
    ir[0, 0] = max([region[0,0], ir[0, 0]])
    ir[0, 1] = min([region[0,1], ir[0, 1]])
    return region, s, ir

def ppsvd(x, mpc):
    u, s, v = np.linalg.svd(x)
    dpc = []
    exp = []
    dp = []
    ex = []
    for i in range(0, mpc):
        y = np.dot(np.dot(u[:, 0:i+1], np.diag(s[0:i+1])), v[0:i+1, :])
        exp.append(np.linalg.norm(y)/np.linalg.norm(x))
        dpc.append(s[i]/s[i+1])
        dp.append(s[i])
        ex.append(np.linalg.norm(y))
    pc_dpc = [i for i, v in enumerate(dpc) if v >= 2]
    pc_exp = [i for i, v in enumerate(exp) if v >= 0.99]
    if len(pc_dpc) and len(pc_exp):
        if pc_dpc[-1] == pc_exp[0]:
            pcs = [pc_dpc[-1]+1]
        elif pc_dpc[-1] > pc_exp[0]:
            pcs = range(pc_exp[0]+1, pc_dpc[-1]+2)
        else:
            pcs = range(pc_dpc[-1]+1, pc_exp[0]+2)
    elif len(pc_dpc) and not len(pc_exp):
        pcs = [pc_dpc[-1]+1]
    elif not len(pc_dpc) and len(pc_exp):
        pcs = [pc_exp[0]+1]
    else:
        pcs = 1
    return pcs

def polyfitting(c, dtz, cols, ms):
    if len(cols):
        Z = c*ms
        if not len(cols):
            return np.zeros(c.shape), 0, 0
        k = np.polyfit(np.sum(Z[:, cols], axis=1), np.sum(dtz[:, cols], axis=1), 1)
        dd = abs(k[0])*Z
        area = np.sum(dd)
        heig = np.max(np.sum(dd, 0))
        chrom = np.sum(dd, axis=1)
        return chrom, area, heig
    elif np.any(c):
        cc = c/max(c)
        apex = np.argmax(c)
        orgmass = dtz[apex, :]
        k = np.polyfit(ms[0,:], orgmass, 1)
        Z = cc*ms
        dd = abs(k[0])*Z
        area = np.sum(dd)
        heig = np.max(np.sum(dd, 0))
        chrom = np.sum(dd, axis=1)
        return chrom, area, heig
    else:
        return np.zeros(c.shape[0]), 0, 0

def mars(ncr, msrt, options):
    rts = msrt['rt']
    mss = msrt['ms']
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    chrs = []
    orcs = []
    segs = np.zeros((len(rts), 2))
    tzs = np.zeros((len(rts), 2))
    areas = np.zeros(len(rts))
    highs = np.zeros(len(rts))
    for i in range(0, len(rts)):
        print(i)
        if i == 1:
            cc = i
        ms = np.array(mss[i], ndmin=2)
        tz, seg, x, segx, maxpos, irr = findTZ(ncr, rts[i], ms, pw, w, thre)
        c, vsel = gridsearch(x, maxpos, 5, irr)
        chrom, a, h = polyfitting(c, x, vsel, ms)
        chrs.append(chrom)
        orcs.append(np.sum(segx, axis=1))
        segs[i, :] = seg
        tzs[i, :] = tz
        areas[i] = a
        highs[i] = h
    return {'chrs': chrs, 'segs': segs, 'tzs': tzs, 'areas': areas, 'highs': highs, 'rts':rts, 'orc': orcs}

def figplot(x, maxpos):
    ax = plt.figure()
    for ind, val in enumerate(range(maxpos-4, maxpos+5)):
        for jnd in range(1, 8):
            k = 7*ind+jnd
            ax.add_subplot(9,7,k)
            c = ittfa(x, val, jnd)
            plt.plot(c)
    ax.show()
    return ax

def ittfaC(x, maxpos, poss, com):
    C = []
    for ind, val in enumerate(range(maxpos-poss/2, maxpos+poss/2+1)):
        cc = []
        for jnd in range(1, com+1):
            c = ittfa(x, val, jnd)
            cc.append(c)
        C.append(cc)
    return C

def ittfa(d, needle, pcs):
    u, s, v = np.linalg.svd(d)
    t = np.dot(u[:,0:pcs], np.diag(s[0:pcs]))
    row = d.shape[0]
    cin = np.zeros((row, 1))
    cin[needle-1] = 1
    out = cin
    for i in range(0, 100):
        vec = out
        out = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), out)
        out[out < 0] = 0
        out = unimod(out, 1.1, 0)
        out = out/norm(out)
        kes = norm(out-vec)
        if kes < 1e-6 or iter == 99:
            break
    return out

def local_max(c, shift=1):
    whe = np.zeros(c.shape)==0
    for i in range(1,shift+1):
        c_plus = c-np.hstack((c[0:i], c[0:-(i)]))
        c_minu = c-np.hstack((c[i:], c[-(i):]))
        whe = whe & ((c_plus >= 0) & (c_minu >= 0))
        if all(whe == False):
            break
    return whe

def pc_estimate(x, mpc):
    u, s, v = np.linalg.svd(x)
    maxc = min([mpc, x.shape[0]])
    pcs = []
    for i in range(0, maxc):
        y = np.dot(np.dot(u[:, 0:i+1], np.diag(s[0:i+1])), v[0:i+1, :])
        if np.linalg.norm(y)/np.linalg.norm(x) >= 0.999:
            pcs.append(i+1)
            break
        if np.linalg.norm(y)/np.linalg.norm(x) >= 0.99:
            pcs.append(i+1)
    return pcs

def gridsearch(x, col, row0, irr):
    u, s, v = np.linalg.svd(x)
    row = range(1, row0+1)
    pos = []
    vsels = []
    vsels99 = []
    C = []
    nums99 = []
    nums = []
    exp = []
    for i, row0 in enumerate(row):
        c = ittfa(x, col, row0)
        vsel90, vsel99 = count99(x, c)
        y = np.dot(np.dot(u[:, 0:i + 1], np.diag(s[0:i + 1])), v[0:i + 1, :])
        exp.append(np.linalg.norm(y) / np.linalg.norm(x))
        vsels.append(vsel90)
        vsels99.append(vsel99)
        nums.append(len(vsel90))
        nums99.append(len(vsel99))
        pos.append(np.argmax(c)+1)
        C.append(c)
    op1 = np.where(np.array(exp) >= 0.99)[0]
    op2 = np.where((np.array(pos) >= irr[0]) & (np.array(pos) < irr[1]))[0]
    op = [val for val in op1 if val in op2]
    if len(op):
        n99 = np.array(nums99)[op]
        n90 = np.array(nums)[op]
        if np.any(n99):
            if max(n99) >= 10:
                mp = np.where(n99 >= 10)[0]
                c = C[op[mp[-1]]]
                vsel = vsels[op[mp[-1]]]
                return c, vsel
            pp = op[np.where(n99 == max(n99))[0][-1]]
            c = C[pp]
            vsel = vsels[pp]
            return c, vsel
        elif np.any(n90):
            pp = op[np.where(n90 == max(n90))[0][0]]
            c = C[pp]
            vsel = vsels[pp]
            return c, vsel
        else:
            return np.zeros((x.shape[0], 1)), []
    else:
        return np.zeros((x.shape[0], 1)), []


def count99(x, c):
    vse = np.where(np.any(x, axis=0))[0]
    cc = np.hstack((x[:, vse], c))
    cors = np.corrcoef(cc.T)[0:-1, -1]
    # cors[np.where(cors != cors)[0]] = 0
    vsel90 = vse[np.where(cors >= 0.9)[0]]
    vsel95 = vse[np.where(cors >= 0.99)[0]]
    return vsel90, vsel95

# def count99(x, c0):
#     c = c0[:, 0]
#     vse = np.where(np.any(x, axis=0))[0]
#     SI = []
#     for i in vse:
#         xx = max(c)*x[:, i]/max(x[:, i])
#         si = 1-np.sum(np.abs(xx-c))/np.sum(xx+c)
#         SI.append(si)
#     vsel90 = vse[np.where(np.array(SI) >= 0.90)[0]]
#     vsel95 = vse[np.where(np.array(SI) >= 0.95)[0]]
#     # plt.plot(x[:, vsel95])
#     # plt.show()
#     return vsel90, vsel95

if __name__ == '__main__':
    pkl_file = open('heye31-6.pkl')
    # pkl_file = open('LMJ3')
    # pkl_file = open('test')
    # pkl_file = open('C:\Users\Administrator\Desktop\li/MARS_Project.pkl')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:/MARS/HEYE/31/31f.cdf"
    # fn = "E:\metadata\heye\20170420\.CDF"
    # fn = "E:\BDY-Synchronize\pycharm-GUI\pycharm_project\storedata\LMJ/71.CDF"
    # fn = "E:\BDY-Synchronize\pycharm-GUI\pycharm_project\storedata\LMJ/75.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']

    com = 172
    tz, seg, x, segx, maxpos, irr = findTZ(ncr, rts[com], np.array(ms[com], ndmin=2), pw, w, thre)
    pcs = pc_estimate(x, 5)
    c, vsel = gridsearch(x, maxpos, 5, irr)
    c0 = ittfa(x, maxpos, 1)
    c1 = ittfa(x, maxpos, 2)
    c2 = ittfa(x, maxpos, 3)
    c3 = ittfa(x, maxpos, 4)
    c4 = ittfa(x, maxpos, 5)

    fig1 = plt.figure(311)
    plt.subplot(311)
    plt.plot(x)
    plt.subplot(312)
    plt.plot(c0, 'g')
    plt.plot(c1, 'r')
    plt.plot(c2, 'y')
    plt.plot(c3, 'c')
    plt.plot(c4, 'k')
    chrom, area, heig = polyfitting(c, x, vsel, np.array(ms[com], ndmin=2))
    plt.subplot(313)
    plt.plot(np.sum(x, axis=1))
    plt.plot(chrom, 'r')
    plt.show()