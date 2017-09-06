__author__ = 'Administrator'

import scipy.io.netcdf as nc
from numpy import zeros, hstack
import numpy as np
import math
import sys
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import figure, show, plot, text
from NetCDF import netcdf_reader


def svdX(X):
    u, s, v = np.linalg.svd(X)
    T = np.dot(u,s)
    return u, T,

def FR(x, s, o, com):
    z = range(0, min([min(s), min(o)]))+range(max([max(s),max(o)]), x.shape[0])
    xs = x[s,:]
    xz = x[z,:]
    xo = x[o,:]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))
    u, s0, v = np.linalg.svd(xc)
    t = np.dot(u[:,0:com],np.diag(s0[0:com]))
    # t = np.dot(u,np.diag(s0))

    # t = np.dot(u, np.diag(s))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T,t)),t.T),np.sum(mc,1))
    u1,s1,v1 = np.linalg.svd(x)
    t1 = np.dot(u1[:,0:com],np.diag(s1[0:com]))
    # t1 = np.dot(u1, np.diag(s1))
    c = np.dot(t1,r)

    ind = np.argmax(np.abs(c[s]))
    if c[s][ind]<0:
        c = -c
    c[c<0]=0


    # plot(cc)
    # show()
    # if max(o)>max(s):
    #     C1 = np.vstack((x[0:min(o),:], cc))
    #     C = np.vstack((C1, np.zeros(x[max(o):x.shape[0],:].shape)))
    # else:
    #     C1 = np.vstack((np.zeros(x[0:min(o),:].shape), cc))
    #     C = np.vstack((C1, x[max(o):x.shape[0],:].shape))
    # C[C<0]=0

    spec = x[s[ind],:]
    cc = c/c[s[ind]]
    res_x = np.dot(np.array(cc,ndmin=2).T, np.array(spec, ndmin=2))
    xx = x - res_x


    # plot(res_x)
    # show()
    # C[C > 0] = 0
    # xx = x + C
    return cc, xx


def plotScore(u):
    plot(u[:,0], u[:,1], 'ro')
    C = np.arange(0, u.shape[0])
    for a, b, c in zip(u[:,0], u[:,1], C):
        text(a, b + 0.001, '%.0f' % c, ha='center', va='bottom', fontsize=7)
    show()

def FSWFA(x, w, mpc):
    unit = int(np.floor(w / 2))
    L = range(unit, x.shape[0]-unit)
    em = np.zeros((x.shape[0]-w+1, mpc))
    for j, v in enumerate(L):
        sx = x[j:j+w,:]
        u, s, v = np.linalg.svd(np.dot(sx,sx.T))
        em[j, :] = np.sqrt(s[0:mpc])#np.log10(np.sqrt(s[0:mpc])) #
    return L, em


if __name__ == '__main__':

    filename ="F:\MARS\data_save\D6/73.CDF"
    ncr = netcdf_reader(filename, bmmap=False)
    tic = ncr.tic()

    m = ncr.mat(1830, 1910, 1)
    plot(m['d'])
    show()

    s = range(22, 33)
    z = range(33, 40)
    c, xx = FR(m['d'], s, z, 4)
    plot(c)
    show()
    plot(xx)
    show()

    # m = ncr.mat(3795, 3825, 1)
    # plot(m['d'])
    # show()
    #
    # s = range(0, 7)
    # z = range(7, 27)
    # c, xx = FR(m['d'], s, z, 3)
    # plot(c)
    # show()
    # plot(xx)
    # show()

    # u, T = svdX(m['d'])
    # plotScore(u)
    #
    #
    # l, em = FSWFA(m['d'], 7, 3)
    # plot(l, em)
    # a = np.arange(0, len(l))
