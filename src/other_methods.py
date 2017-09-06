__author__ = 'Administrator'

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


class GCMSReader:
    def __init__(self, fname, bmmap=True):
        self.filename = fname
        self.f = nc.netcdf.netcdf_file(fname, 'r', mmap=bmmap)
        self.mass_values = self.f.variables['mass_values']
        self.intensity_values = self.f.variables['intensity_values']
        self.scan_index = self.f.variables['scan_index']
        self.total_intensity = self.f.variables['total_intensity']
        self.scan_acquisition_time = self.f.variables['scan_acquisition_time']
        self.mass_max = np.max(self.f.variables['mass_range_max'].data)
        self.mass_min = np.max(self.f.variables['mass_range_min'].data)
        if sys.byteorder == "little":
            self.nbo = "<"
        else:
            self.nbo = ">"

    def mz_point(self, n):
        scan_index_end = hstack((self.scan_index.data, np.array([len(self.intensity_values.data)], dtype=int)))
        ms = {}
        inds = range(scan_index_end[n], scan_index_end[n + 1])
        ms['mz'] = self.mass_values[inds]
        ms['val'] = self.intensity_values[inds]
        return ms

    def mz_rt(self, t):
        scan_index_end = hstack((self.scan_index.data, np.array([len(self.intensity_values.data)], dtype=int)))
        ms = {}
        tic_dict = self.tic()
        rt = tic_dict['rt']
        n = np.searchsorted(rt, t)
        inds = range(scan_index_end[n], scan_index_end[n + 1])
        ms['mz'] = self.mass_values[inds]
        ms['val'] = self.intensity_values[inds]
        return ms

    def tic(self):
        tic_dict = {'rt': self.scan_acquisition_time.data / 60.0, 'val': self.total_intensity.data}
        if tic_dict['val'].dtype.byteorder != self.nbo:
            tic_dict['val'] = tic_dict['val'].byteswap().newbyteorder()
        return tic_dict

    def mat_rt(self, rt_start, rt_end):
        indmin, indmax = np.searchsorted(self.tic()['rt'], (rt_start, rt_end))
        rt = self.tic()['rt'][indmin:indmax + 1]
        mass_max = np.max(self.f.variables['mass_range_max'].data)
        mass_min = np.max(self.f.variables['mass_range_min'].data)
        mz = np.linspace(mass_min, mass_max, num=mass_max - mass_min + 1)
        return {'mat': self.mat(indmin, indmax)['mat'], 'rt': rt, 'mz': mz}

    def mat_list(self, peaks):
        f = nc.netcdf_file(self.filename, 'r', mmap=True)
        mass_values = f.variables['mass_values']
        intensity_values = f.variables['intensity_values']
        scan_index = f.variables['scan_index']
        scan_index_end = np.hstack((scan_index.data, np.array([len(intensity_values.data)], dtype=int)))
        mass_max = np.max(f.variables['mass_range_max'].data)
        mass_min = np.max(f.variables['mass_range_min'].data)
        mz = np.linspace(mass_min, mass_max, num=mass_max - mass_min + 1)
        rg = np.linspace(mass_min - 0.5, mass_max + 0.5, num=mass_max - mass_min + 2)
        c = len(peaks)
        r = int(mass_max - mass_min + 1)
        mo = np.zeros((r, c, 10))
        for j in range(c):
            mz_val = mass_values[scan_index_end[peaks[j]]:scan_index_end[peaks[j] + 1]]
            ms = intensity_values[scan_index_end[peaks[j]]:scan_index_end[peaks[j] + 1]]
            inds = np.searchsorted(mz_val, rg)
            for i in range(0, r):
                mo[i, j, 0:(inds[i + 1] - inds[i])] = ms[inds[i]:inds[i + 1]]
        return {'mat': np.sum(mo, 2).T, 'mz': mz}

    def mat(self, a, bin):
        f = nc.netcdf_file(self.filename, 'r', mmap=False)
        t = f.variables['scan_acquisition_time'].data / 60.0
        mass_values = f.variables['mass_values']
        intensity_values = f.variables['intensity_values']
        scan_index = f.variables['scan_index']
        scan_index_end = np.hstack((scan_index.data, np.array([len(intensity_values.data)], dtype=int)))
        mass_max = np.max(f.variables['mass_range_max'].data)
        mass_min = np.max(f.variables['mass_range_min'].data)
        mz = np.linspace(mass_min, mass_max, num=mass_max - mass_min + 1)
        if a[0] < 0 and a[1] > len(t):
            print('please print suitable index of retention time')
            exit()
        elif a == 'None':
            imin = 0
            imax = int(len(t))
        else:
            imin = a[0]
            imax = a[1]
        rt = t[imin:imax+1]
        c = int(imax - imin)
        r = int(mass_max - mass_min + 1)
        mo = np.zeros((c, r))
        for j in range(imin, imax):
            msnext = scan_index_end[j+1]
            if msnext > np.shape(mass_values):
                mo[j:imax, :] = []
                break
            else:
                mz_val = mass_values[scan_index_end[j]:scan_index_end[j + 1]]
                sp = intensity_values[scan_index_end[j]:scan_index_end[j + 1]]
                ind = np.round((mz_val - mass_min) / bin)
                position = np.nonzero(mz_val > np.max(mz))
                ind = np.delete(ind, position)
                sp = np.delete(sp, position)
                ind2 = np.unique(ind)

                if np.shape(ind2) != np.shape(ind):
                    sp2 = np.zeros(ind2.shape[0])
                    for i in range(0, ind2.shape[0]-1):
                        tempind = np.nonzero(ind == ind2[i])
                        sp2[i] = np.sum(sp[tempind[0]])
                else:
                    sp2 = sp
                    ind2 = ind
                mo[j-imin, np.array(ind2, dtype=np.int32)] = sp2
        return{'d': mo, 'rt': rt, 'mz': mz}

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

def backremv(x, start1, end1, start2, end2):
    mn = x.shape
    bak2 = np.zeros(mn)
    for i in range(0, mn[1]-1):
        tiab = np.append(x[start1:end1, i], x[start2:end2, i])
        reg = np.append(np.arange(start1, end1), np.arange(start2, end2))
        rm = reg-reg.mean()
        tm = tiab-tiab.mean()
        b = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
        s = tiab.mean()-np.dot(reg.mean(), b)
        b_est = s+b*np.arange(mn[0])
        bak2[:, i] = x[:, i]-b_est
    return{'Back': bak2}

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

def mcr_als(d, ns, crs, nit):
    mod = [1.1, 2]
    tolsigma = 0.1
    niter = -1
    idev = 0
    if len(crs.shape) == 1:
        crs.reshape((crs.shape[0], 1))
    if d.shape[0] == crs.shape[0]:
        conc = crs
        spec = np.dot(np.dot(np.linalg.inv(np.dot(conc.T, conc)), conc.T), d)
    elif d.shape[0] == crs.shape[1]:
        conc = crs.T
        spec = np.dot(np.dot(np.linalg.inv(np.dot(conc.T, conc)), conc.T), d)
    else:
        print('please import right initial estimation')
        exit()
    dn = d
    u, s, v, d, sd = pcarep(dn, ns)
    sstn = np.sum(np.power(dn, 2))
    sst = np.sum(np.power(d, 2))
    sigma2 = np.power(sstn, 0.5)

    while niter < nit:
        niter += 1
        conc = np.dot(np.dot(d, spec.T), np.linalg.inv(np.dot(spec, spec.T)))
    # non-negative for concentration profile
        conc2 = np.copy(conc)
        for j in range(0, conc.shape[0]):
            a = fnnls(spec, d[j, :], tole='None')
            conc2[j, :] = a['xx']
        conc = conc2
    # unimodality  for concentration profile
        conc2 = np.copy(conc)
        conc2 = unimod(conc2, mod[0], mod[1])
        conc = conc2
    # non-negative for mass spectrum
        spec = np.dot(np.dot(np.linalg.inv(np.dot(conc.T, conc)), conc.T), d)
        spec2 = spec
        for j in range(0, spec.shape[1]):
            a = fnnls(conc.T, d[:, j], tole='None')
            spec2[:, j] = a['xx']
        spec = spec2

        res = d-np.dot(conc, spec)
        resn = dn-np.dot(conc, spec)
        u = np.sum((np.power(res, 2)))
        un = np.sum((np.power(resn, 2)))
        sigma = np.power(u / (d.shape[0]*d.shape[1]), 0.5)
        change = (sigma2 - sigma)/sigma
        if change < 0.0:
            idev += 1
        else:
            idev = 0
        change = np.dot(100, change)
        lof_pca = np.power((u/sst), 0.5)*100
        lof_exp = np.power((un/sstn), 0.5)*100
        r2 = (sstn-un)/sstn
        if change > 0 or niter == 0:
            sigma2 = sigma
            copt = conc
            sopt = spec
            itopt = niter+1
            sdopt = np.array([lof_pca, lof_exp])
            ropt = res
            r2opt = r2
            # Fitting error (lack of fit, lof) in % (PCA) = 1.0606
            # Fitting error (lack of fit, lof) in % (exp) = 4.2451
            # Percent of variance explained (r2) is 99.8198
            # Optimum in the iteration 5
        if abs(change) < tolsigma:
            print('CONVERGENCE IS ACHIEVED, STOP!!!')
            break
        if idev >= 20:
            print('FIT NOT IMPROVING FOR 20 TMES CONSECUTIVELY (DIVERGENCE?), STOP!!!')
            break
    return {'copt': copt, 'sopt': sopt, 'sdopt': sdopt, 'r2opt': r2opt, 'ropt': ropt, 'itopt': itopt}

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

def pcarep(xi, nf):
    u, s, v = np.linalg.svd(xi)
    x = np.dot(np.dot(u[:, 0:nf], np.diag(s[0:nf])), v[0:nf, :])
    res = xi - x
    sst1 = np.power(res, 2).sum()
    sst2 = np.power(xi, 2).sum()
    sigma = np.power(sst1/sst2, 0.5)*100
    return u, s, v, x, sigma

def frr(data, sel, zer, pc):
    x = data['d']
    x_sel = x[sel[0]:sel[1], :]
    x_zer = x[range(zer[0], zer[1])+range(zer[2], zer[3]), :]
    x_con = np.vstack((x_zer, x_sel))
    c_zer = np.zeros(x_zer.shape)
    x_m = np.vstack((c_zer, x_sel))
    u2, s2, v2 = np.linalg.svd(x_con)
    t = np.dot(u2[:, 0:pc], np.diag(s2[0:pc]))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), x_m)

    u1, s1, v1 = np.linalg.svd(x)
    x_ext = np.dot(np.dot(u1[:, 0:pc], np.diag(s1[0:pc])), r)
    #ind = np.argsort(chr.sum(axis=0))
    #C = chr[:, ind]
    #S = np.dot(np.dot(np.linalg.inv(np.dot(chr.T, chr)), chr.T), x)
    x_lef = x-x_ext
    return{'x_ext': x_ext, 'x_lef': x_lef, 'rt': data['rt']}

def findTZ(data, MSRT, w, thre):
    d = data['d']
    rto = data['rt']
    ms = MSRT['MS']
    # rt0 = MSRT['RT']
    max_pc = 5
    resp, eig = MSWFA(d, ms, max_pc, w, thre)
    if resp.size:
        tz = rto[resp[0], resp[1]]
    else:
        tz = np.zeros(1, 2)
    return tz

def MSWFA(d, ms, max_pc, w, thres):
    s = np.zeros(d.shape[0], 1)
    for j in range(0, d.shape[0]-w+1-1):
        d1 = d[j:j+w-1, :]
        u, s, e = np.linalg.svd(d1)
        M = np.dot(np.dot(np.dot(e[:, 0:max_pc-1].T, ms), ms.T), e[:, 0:max_pc-1])
        S = np.diag(M)
        s[j+math.ceil((w-1)/2)] = np.max(S)
    vsel = np.nonzero(s >= thres)
    vse = s >= thres
    if vsel.size:
        b = np.ceil(d.shape[1]/2)
        f = abs(b-vsel)
        p = np.nonzero(f == np.min(f))
        start = np.min(np.nonzero(vse[0:vsel[p]] == 1))
        end = vsel[p] + np.max(np.nonzero(vse[vsel[p]:] == 1))
        region = np.array([start, end])
    else:
        region = np.array([])
    return{'reg': region, 'eig': s}

class RESOLUTION:
    def __init__(self, data, tz, MS):
        self.data = data
        self.MS = MS
        self.tz = tz
        slice = np.searchsorted(data.T, tz)
        self.dataslice = data.X[:, slice[0]:slice[1]]

    def NEEDLE(self):
        tzs = range(self.TZ[0], self.TZ[1], self.f / 60)
        coefs = []
        for i in range(0, self.dataslice.shape[1]):
            coefs.append(np.corrcoef(self.MS, self.dataslice[:, i]))
        val, pos = np.max(coefs)
        needles = tzs[pos]
        return{'NEE': needles}

    def PCS(self, max_pc):
        x = self.dataslice
        d = []
        u, s, v = np.linalg.svd(x)
        r1 = norm(x - u[:, 0]*s[0, 0]*v[:, 0]) / norm(x)
        for j in range(0, np.min([max_pc, x.shape[1]])-1):
            e = u[:, 0:j]
            f = x[x.pure(nr=j, f=0.01)['IMP'], :]
            d.append(j-np.trace(np.dot(np.dot(np.dot(np.transpose(e), f), np.transpose(f)), e)))
        dv, index = np.sort(d)
        if dv[0] > 0.1:
            pcs = 0
        elif index[0] == 1 and r1 < 0.05:
            pcs = 1
        elif index[0] == 1 and r1 >= 0.05:
            pcs = index[1]
        else:
            pcs = index[0]
        return{'PCS': pcs}

    def pure(self, nr, f):
        d = self.dataslice
        nrow, ncol = d.shape()
        s = np.var(d)
        m = np.mean(d)
        ll = s**2 + m**2
        f = np.max(m) * f / 100
        p = s / (m+f)
        mp, imp = np.max(p)
        l = math.sqrt(s**2 + (m + f) ** 2)
        dl = []
        for j in range(0, ncol-1):
            dl[:, j] = d[:, j] / l[j]
        c = np.dot(np.transpose(dl), dl/nrow)
        w = []
        w[0, :] = ll / (l**2)
        p[0, :] = np.multiply(w[0, :], p[0, :])
        s[0, :] = np.multiply(w[0, :], s[0, :])
        dm = []
        for i in range(1, nr-1):
            for j in range(0, ncol-1):
                dm[0, 0] = c[j, j]
                for k in range(1, i-1):
                    kvar = imp[k-1]
                    dm[0, k] = c[j, kvar]
                    dm[k, 0] = c[kvar, j]
                    for kk in range(1, i-1):
                        kkvar = imp[kk-1]
                        dm[k, kk] = c[kvar, kkvar]
                w[i, j] = np.linalg.det(dm)
                p[i, j] = np.multiply(p[0, j], w[i, j])
                s[i, j] = np.multiply(s[0, j], w[i, j])
            mp[i], imp[i] = np.max(p[i, :])
        sn = []
        for i in range(0, nr-1):
            sn[0:nrow, i] = d[0:nrow, imp[i]]
        ss = np.transpose(sn)
        sp = []
        for bit in range(0, ss.shape(0)):
            sr = math.sqrt(np.cumsum(np.multiply(ss[bit, :], ss[bit, :])))
            sp[bit, :] = np.multiply(ss[bit, :], sr)
        return{'SP': sp, 'IMP': imp}

    def ittfa(self, needle, pcs):
        d = self.dataslice
        u, s, v = np.linalg.svd(d)
        t = np.dot(u, s)
        t = t[:, 0:pcs-1]
        row = d.shape(0)
        cin = np.zeros(row,)
        cin[needle] = 1
        out = cin
        for i in range(0, 499):
            vec = out
            out = np.dot(np.dot(t, np.linalg.pinv(np.dot(np.transpose(t), t))), out)
            out[out < 0] = 0
            out = self.unimod(out, 1.1, 2)
            out = out / norm(out)
            kes = norm(out-vec)
            if kes < 1e-6 or iter == 499:
                nu = iter
                break
        return{'Chr': out, 'Num': nu}

    def unimod(self, c, rmod, cmod):
        ns = c.shape()
        for i in range(0, ns[1]-1):
            imax = c[:i].argmax()
        for j in range(0, ns[0]-1):
            rmax = c[imax[j], j]
            k = imax[j]
            while k > 1:
                k = k-1
                if c[k, j] <= rmax:
                    rmax = c[k, j]
                else:
                    rmax2 = rmax*rmod
                    if c[k, j] > rmax2:
                        if cmod == 0:
                            c[k, j] = 1e-30
                        if cmod == 1:
                            c[k+1, j] = c[k+1, j]
                        if cmod == 2:
                            if rmax > 0:
                                c[k, ] = (c[k, j]+c[k+1, j])/2
                                c[k+1, j] = c[k, j]
                                k = k+2
                            else:
                                c[k, j] = 0
                        rmax = c[k, j]
            rmax = c[imax[j], j]
            k = imax[j]

            while k < ns:
                k = k+1
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
        return{'C': c}

    def polyfitting(self, tz, nee, A, thre1, thre2):
        dtz = self.dataslice[:, tz[1]:tz[2]]
        # CN = dtz.ittfa(needle=nee ,pcs=A)
        c = dtz.ittfa(needle=nee, pcs=A)['Chr']
        col1 = np.nonzero(dtz.sum(1) >= thre1*(dtz.sum(1).max()))
        if tz[1] == 0:
            chrom = []
            area = 0
            heig = 0
        else:
            Coef = []
            for i in range(0, dtz.shape(0)):
                Coef.append(np.corrcoef(c, dtz[i, :])[0, 1])
            col2 = np.nonzero(Coef >= thre2)
            col = [val for val in col1 if val in col2]
            if col.size():
                Z = c*self.MS
                k = np.polyfit(Z[col, :], dtz[col, :], 1)
                dd = k[1]*Z
                area = dd.sum()
                heig = dd.sum(0).max()
                chrom = k[1]*c
            else:
                area = 0
                heig = 0
                chrom = np.zeros(c.shape(1), )
        return{'C': chrom, 'area': area, 'Hei': heig}

def plot_efa(efas, thre):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a = np.arange(1, efas['efl'].shape[1])
    b = np.arange(1, efas['ebl'].shape[1])[::-1]
    ax.plot(a, efas['efl'], 'r')
    ax.plot(b, efas['ebl'], 'b')
    if thre != 'none':
        c = thre*np.ones((1, a.shape[0]))
        ax.plot(a, c.T, 'k', lw=3)
        ax.axis([0, a.shape[0]+1, 0.8*thre, 1.05*max([efas['efl'].max(), efas['ebl'].max()])])


def plot_ms(ms):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.vlines(ms['mz'], zeros((len(ms['mz']),)), ms['val'], color='k', linestyles='solid')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%3.4f'))
    plt.show()

def plot_tic(tic_dict):
    plt.plot(tic_dict['rt'], tic_dict['val'])
    plt.show()



if __name__ == '__main__':

    filename = 'N21.cdf'
    ncr = GCMSReader(filename, True)
    # tic = ncr.tic()
    a = [1750, 1850]
    m = ncr.mat(a, 1.0)

    # pures = pure(m['d'], 2, 0.1)
    # mrcs = mcr_als(m['d'], 2, pures['SP'], 50)

    # fig = plt.figure()
    # plt.plot(m['d'])
    # plt.axis('tight')

    # pures = pure(m['d'], 2, 0.1)
    # fig = plt.figure()
    # plt.plot(pures['SP'].T)
    # plt.axis('tight')

    # efas = efa(m)
    # plot_efa(efas, thre=8)

    # m_n = backremv(m['d'], 0, 20, 70, 100)
    # fig = plt.figure()
    #  ax1 = fig.add_subplot(2, 1, 1)
    #  plt.plot(m['d'])
    #  plt.axis('tight')
    #  ax2 = fig.add_subplot(2, 1, 2)
    #  plt.plot(m_n['Back'])
    #  plt.axis('tight')

    # frrs = frr(m, [32, 43], [0, 25, 54, 90], 2)
    # fig = plt.figure()
    # plt.plot(frrs['x_ext'])
