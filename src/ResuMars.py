__author__ = 'Administrator'
import numpy as np
from scipy.linalg import norm


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
        l = np.power(s**2 + (m + f) ** 2, 0.5)
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
            sr = np.power(np.cumsum(np.multiply(ss[bit, :], ss[bit, :])), 0.5)
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