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
import time

from MARS_methods import PCS, ittfa, MSWFA, mars, findTZ, gridsearch
from NetCDF import netcdf_reader

def count99(x, c):
    vse = np.where(np.any(x, axis=0))[0]
    cc = np.hstack((x[:,vse], c))
    cors = np.corrcoef(cc.T)[0:-1, -1]
    # cors[np.where(cors != cors)[0]] = 0
    vsel90 = vse[np.where(cors >= 0.9)[0]]
    vsel95 = vse[np.where(cors >= 0.99)[0]]
    sic = np.argmax(cors)
    return vsel90, vsel95, sic, len(vsel95)


def polyfitting(c, dtz, cols, ms):
    if len(cols):
        Z = c*ms
        if not len(cols):
            return np.zeros(c.shape), 0, 0
        k = np.polyfit(np.sum(Z[:, cols], axis=1), np.sum(dtz[:, cols], axis=1), 1)
        #cors[np.where(cors != cors)[0]] = 0
        # if len(np.where(k != k)[0]):
        #     return np.zeros(c.shape), 0, 0
        dd = abs(k[0])*Z
        # plt.plot(np.sum(dtz, axis=1), 'go--')
        # plt.plot(np.sum(dd,axis=1), 'rs--')
        # plt.show()
        resn = dtz-dd
        un = np.sum((np.power(resn, 2)))
        sstn = np.sum(np.power(dtz, 2))
        R2 = (sstn-un)/sstn
        area = np.sum(dd)
        heig = np.max(np.sum(dd, 0))
        chrom = np.sum(dd, axis=1)
        return chrom, area, heig,R2
    elif np.any(c):
        cc = c/max(c)
        apex = np.argmax(c)
        orgmass = dtz[apex, :]
        k = np.polyfit(ms[0,:], orgmass, 1)
        Z = cc*ms
        dd = abs(k[0])*Z

        resn = dtz-dd
        un = np.sum((np.power(resn, 2)))
        sstn = np.sum(np.power(dtz, 2))
        R2 = (sstn-un)/sstn

        area = np.sum(dd)
        heig = np.max(np.sum(dd, 0))
        chrom = np.sum(dd, axis=1)

        return chrom, area, heig, R2
    else:
        return np.zeros(c.shape[0]), 0, 0, 0

if __name__ == '__main__':

    pkl_file = open('PD6--RM--')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    # fn = "F:\MARS\data_save\D6/75.CDF"
    options = data['options']
    # ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']
    mz = msrt['mz']
    files = data['files']['files']
    # fn = files[0]
    ncr = netcdf_reader("F:\\MARS\\data_save\\D6/73.CDF", bmmap=False)

    sics = []
    times = []
    pcs = []
    R2 = []
    nums = []

    tt1 = time.time()
    for i, m in enumerate(ms):

        t1 = time.time()
        tz1, seg1, x1, segx1, maxpos1, irr1 = findTZ(ncr, rts[i], np.array(ms[i], ndmin=2), pw, w, mth='MSWFA')
        t2 = time.time()
        dt1 = t2-t1

        pc1 = PCS(x1, 6)

        u, s, v = np.linalg.svd(x1)
        xn = np.dot(np.dot(u[:, 0:pc1], np.diag(s[0:pc1])), v[0:pc1, :])
        resn = x1 - xn
        un = np.sum((np.power(resn, 2)))
        sstn = np.sum(np.power(x1, 2))
        R2.append((sstn - un) / sstn)

        c1 = ittfa(x1, maxpos1, pc1)
        vsel10, vsel11, sic1, nums1 = count99(x1, c1)
        chrom1, area1, heig1,r2 = polyfitting(c1, x1, vsel11, np.array(ms[i], ndmin=2))
        sics.append(mz[sic1])
        pcs.append(pc1)
        times.append(dt1)
        nums.append(nums1)

    tt2 = time.time()
    print(tt2-tt1)
    result = np.array([sics, pcs, R2, times, nums])
    np.savetxt('F:\MARS\M1.txt', result, delimiter=",", fmt="%s")

    # sics = []
    # times = []
    # pcs = []
    # R2 = []
    # nums = []
    #
    # tt1 = time.time()
    # for i, m in enumerate(ms):
    #
    #     t1 = time.time()
    #     tz1, seg1, x1, segx1, maxpos1, irr1 = findTZ(ncr, rts[i], np.array(ms[i], ndmin=2), pw, w, mth='RM')
    #     t2 = time.time()
    #     dt1 = t2 - t1
    #
    #     c1, vsel, pc1 = gridsearch(x1, maxpos1, 6, irr1)
    #     u, s, v = np.linalg.svd(x1)
    #     xn = np.dot(np.dot(u[:, 0:pc1], np.diag(s[0:pc1])), v[0:pc1, :])
    #     resn = x1 - xn
    #     un = np.sum((np.power(resn, 2)))
    #     sstn = np.sum(np.power(x1, 2))
    #     R2.append((sstn - un) / sstn)
    #
    #     vsel10, vsel11, sic1, nums1 = count99(x1, c1)
    #     chrom1, area1, heig1, r2 = polyfitting(c1, x1, vsel11, np.array(ms[i], ndmin=2))
    #     sics.append(mz[sic1])
    #     pcs.append(pc1)
    #     times.append(dt1)
    #     nums.append(nums1)
    #
    #     print(i)
    #
    # tt2 = time.time()
    # print(tt2 - tt1)
    # result = np.array([sics, pcs, R2, times, nums])
    # np.savetxt('F:\MARS\M2.txt', result, delimiter=",", fmt="%s")