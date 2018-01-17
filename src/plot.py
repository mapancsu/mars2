from __future__ import division
__author__ = 'Administrator'
'''
MARS (Massspectrum-Assited Resolution of Signal)

Created on 2015-11-10

@author: pandemar

'''

from numpy.linalg import norm
from scipy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from NetCDF import netcdf_reader
from MARS_methods import findTZ, MSWFA, gridsearch, polyfitting, ittfa, count99
from chemoMethods import backremv


def gauss(pos, sigma, lengths, high):
    rang = np.array(range(0, lengths))
    t = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (rang - pos) ** 2 / (2 * sigma ** 2))
    t = t/max(t)*high
    return rang, t

def plot_simulate():
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 1000)
    fig = plt.figure()

    rang, t1 = gauss(50, 5, 100, 1)
    plt.subplot(231)
    plt.plot(t1)
    # plt.tick_params(axis='both', labelsize=15)

    rang, t2 = gauss(50, 10, 100, 0.5)
    plt.subplot(234)
    plt.plot(t1)
    plt.plot(t2)
    # plt.tick_params(axis='both', labelsize=15)

    rang, t3 = gauss(38, 10, 100, 1)
    rang, t4 = gauss(62, 10, 100, 0.8)
    plt.subplot(232)
    plt.plot(t3)
    plt.plot(t4)
    # plt.tick_params(axis='both', labelsize=15)

    rang, t5 = gauss(50, 10, 100, 1)
    rang, t6 = gauss(43, 4, 100, 0.4)
    plt.subplot(233)
    plt.plot(t5)
    plt.plot(t6)
    # plt.tick_params(axis='both', labelsize=15)

    rang, t7 = gauss(35, 8, 100, 0.9)
    rang, t8 = gauss(50, 8, 100, 0.85)
    rang, t9 = gauss(65, 8, 100, 0.9)
    plt.subplot(235)
    plt.plot(t7)
    plt.plot(t8)
    plt.plot(t9)
    # plt.tick_params(axis='both', labelsize=15)
    plt.show()

def plot_RM():
    pkl_file = open('PD6--RM--')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:\MARS\data_save\D6/73.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']
    mz = msrt['mz']
    com = 33

    rttic = ncr.tic()['rt']

    tz, seg1, x, segx, maxpos, irr = findTZ(ncr, rts[com], np.array(ms[com], ndmin=2), pw, w, mth='MSWFA')
    resp1, eig1, ir = MSWFA(segx, np.array(ms[com], ndmin=2), 5, w, mth='MSWFA')

    tz, seg2, x, segx, maxpos, irr = findTZ(ncr, rts[com], np.array(ms[com], ndmin=2), pw, w, mth='RM')
    resp2, eig2, ir = MSWFA(segx, np.array(ms[com], ndmin=2), 5, w, mth='RM')

    po = rttic[tz]
    x1 = rttic[range(seg1[0], seg1[1])]
    x2 = rttic[range(seg2[0], seg2[1])]
    pos = np.searchsorted(x2, po)

    plt.figure()
    ax1 = plt.subplot(211)

    ax1.plot(x1, eig1, color = 'r', label = 'MSWFA')
    ax1.plot(x2, eig2, color = 'b', label = 'RM')

    ax1.scatter(np.array(po), np.array(eig2)[pos], s=45, marker='^',
            color='black')
    maxpos = np.argmax(eig2)
    maxrt = rttic[np.arange(seg1[0], seg1[1])[maxpos]]
    ax1.scatter(np.array([maxrt]), np.array(eig2)[maxpos], s=45, marker='o',
            color='blue')
    ax1.set_xlim((min(x1), max(x1)))
    ax1.set_ylim((0, 1.02))
    plt.subplots_adjust(bottom=0.1, top=0.95, hspace=0.25)

    ax1.set_ylabel('score')
    ax1.set_xlabel('retention time (min)')
    plt.legend(loc='upper left')

    ax = plt.subplot(212)
    ii = 12
    y1 = x[ii,:]/norm(x[ii,:])
    ax.vlines(mz, np.zeros((len(mz),)), y1, color='b', linestyles='solid', label='R')

    y2 = ms[com]/norm(ms[com])
    ind = np.argsort(y2)[::-1][1:20]
    yy = y2[ind]
    mz1 = mz[ind]
    ax.vlines(mz1, np.zeros((len(mz1),)), -yy, color='r', linestyles='solid', label='E')
    ax.plot(range(0, len(mz)), np.zeros(len(mz)), color='k')

    ax.set_xlim((min(mz), 350))
    ax.set_ylim((-0.25, 0.25))

    ax.set_xlabel('m/z')
    ax.set_ylabel('intensity')
    plt.legend(loc='lower left')
    plt.show()


def plot_compara1():
    pkl_file = open('PD6--RM--')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:\MARS\data_save\D6/75.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']
    mz = msrt['mz']
    # com = [79]
    com = range(22, 27)

    rttic = ncr.tic()['rt']

    plt.figure()
    ax = plt.subplot(111)

    rang = []
    colors = ['r','g','c','y','m']
    for ii, i in enumerate(com):
        tz, seg2, x, segx, maxpos, irr = findTZ(ncr, rts[i], np.array(ms[i], ndmin=2), pw, w, mth='RM')
        rang.extend(tz)
        c, vsel,pc = gridsearch(x, maxpos, 6, irr)
        chrom, area, heig,d = polyfitting(c, x, vsel, np.array(ms[i], ndmin=2))
        # plt.subplot(111)
        # plt.plot(np.sum(x, axis=1))
        plt.plot(rttic[range(tz[0], tz[1])], chrom, color=colors[ii], linewidth=3.0,label='c'+str(ii+1))
    plt.plot(rttic[range(tz[0], tz[1])], chrom, color='m',)
    segx = ncr.mat(min(rang), max(rang)-1, 1)
    rtseg = rttic[range(min(rang), max(rang))]
    plt.plot(rtseg, np.sum(segx['d'], axis=1), color='b', linewidth=3.0, label='raw')
    ax.set_xlim((min(rtseg), max(rtseg)))
    ax.set_xlabel('retention time (min)')
    ax.set_ylabel('intensity')
    # plt.legend(loc='upper left')
    plt.show()

def plot_compara2():
    pkl_file = open('PD6--RM--')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:\MARS\data_save\D6/73.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']
    mz = msrt['mz']
    com = 23

    rttic = ncr.tic()['rt']

    plt.figure()
    ax = plt.subplot(111)

    colors = ['r','g','c','y','m']
    tz, seg2, x, segx, maxpos, irr = findTZ(ncr, rts[com], np.array(ms[com], ndmin=2), pw, w, mth='RM')
    # c1, vsel1 = gridsearch(x, maxpos, 6, irr)
    c1 = ittfa(x, maxpos, 3)
    vsel2, vsel99 = count99(x, c1)
    chrom1, area, heig,d = polyfitting(c1, x, vsel2, np.array(ms[com], ndmin=2))
    print(len(vsel99))

    # c2 = ittfa(x, maxpos, 2)
    c2, vsel1,pc = gridsearch(x, maxpos, 6, irr)
    vsel2, vsel99 = count99(x, c2)
    chrom2, area, heig,d = polyfitting(c2, x, vsel2, np.array(ms[com], ndmin=2))
    print(len(vsel99))

    plt.plot(rttic[range(tz[0], tz[1])], chrom1, color=colors[0], linewidth=3.0, label='com number ='+str(3))
    plt.plot(rttic[range(tz[0], tz[1])], chrom2, color=colors[1], linewidth=3.0, label='com number ='+str(4))
    plt.plot(rttic[range(tz[0], tz[1])], np.sum(x, axis=1), color='b', linewidth=3.0, label='raw')
    ax.set_xlim((min(rttic[range(tz[0], tz[1])]), max(rttic[range(tz[0], tz[1])])))
    ax.set_xlabel('retention time (min)')
    ax.set_ylabel('intensity')
    plt.legend(loc='upper left')
    plt.show()

def plot_fig5():
    pkl_file = open('PD6--RM--')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:\MARS\data_save\D6/73.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']
    mz = msrt['mz']
    com = 23

    d = ncr.mat(1845,1895,1)
    rttic = ncr.tic()['rt']
    rts = rttic[range(1845, 1896)]
    pos = np.searchsorted(rts, [12.753])
    plt.plot(d['d'][pos[0], :])
    plt.show()
    map = np.argsort(d['d'][pos[0], :])[::-1][0:10]
    print(map)

    mzs = d['mz']
    mz1 = mzs[123]
    mz2 = mzs[265]
    mz3 = mzs[170]
    ax1 = plt.subplot(211)
    plt.plot(rts, d['d'], color='k', linewidth=1.5, alpha=0.35)
    plt.plot(rts, d['d'][:, 123], color='k', linewidth=1.5, alpha=0.35, label='raw')
    plt.plot(rts, d['d'][:, 123], color='g', label='m/z='+str(int(mz1)))
    plt.plot(rts, d['d'][:, 265], color='r', label='m/z='+str(int(mz2)))
    plt.plot(rts, d['d'][:, 170], color='b', label='m/z='+str(int(mz3)))

    ax1.set_xlim((min(rts), max(rts)))
    ax1.set_xlabel('retention time (min)')
    ax1.set_ylabel('intensity')
    plt.legend(loc='upper right')


    pkl_file = open('standard')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:\MARS\data_save\STDAND/zhi10-5vs1.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)

    d = ncr.mat(1470,1506,1)
    rttic = ncr.tic()['rt']
    rts = rttic[range(1470, 1507)]
    # pos = np.searchsorted(rts, [12.753])
    # plt.plot(d['d'][pos[0], :])
    # plt.show()
    # map = np.argsort(d['d'][pos[0], :])[::-1][0:10]
    # print(map)

    mzs = d['mz']
    ax2 = plt.subplot(212)
    plt.plot(rts,d['d'], color='k', linewidth=1.5, alpha=0.35)

    ax2.set_xlim((min(rts), max(rts)))
    ax2.set_ylim((0, 400000))
    ax2.set_xlabel('retention time (min)')
    ax2.set_ylabel('intensity')
    plt.legend(loc='upper right')

    plt.show()

# def get_R2():
#
#     pkl_file = open('D6-new')
#     data = pickle.load(pkl_file)
#     msrt = data['MSRT']
#     fn = "F:\MARS\data_save\D6/73.CDF"
#     options = data['options']
#     ncr = netcdf_reader(fn, bmmap=False)
#     pw = options['pw']
#     w = options['w']
#     thre = options['thres']
#     rts = msrt['rt']
#     mss = msrt['ms']
#     mz = msrt['mz']
#     segno = msrt['segno']
#     results = data['results']
#
#     inds = [86]
#     tzs = results[0]['tzs'][inds,:]
#     RG = np.arange(np.min(tzs), np.max(tzs))
#     dd = np.zeros((len(RG), len(mz)))
#     xx = ncr.mat(int(RG[0]), int(RG[-1]), 1)
#     for j in inds:
#         newd = np.zeros((len(RG), len(mz)))
#         tz, seg, x, segx, maxpos, irr = findTZ(ncr, rts[j], np.array(mss[j], ndmin=2), pw, w, thre)
#         c, vsel = gridsearch(x, maxpos, 6, irr)
#         chrom, area, heig, d = polyfitting(c, x, vsel, np.array(mss[j], ndmin=2))
#         re = np.searchsorted(RG, [tz[0], tz[1]])
#         newd[re[0]:re[1],:]= d
#         dd = dd+newd
#     resn = xx['d']-dd
#     un = np.sum((np.power(resn, 2)))
#     sstn = np.sum(np.power(xx['d'], 2))
#     R2 = (sstn-un)/sstn
#     print(R2)

def get_R2(bass=None):

    # pkl_file = open('standardn3')
    pkl_file = open('PD6--RM---')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    # fn = "F:\MARS\data_save\STDAND/zhi10-5vs1.CDF"
    fn = "F:\MARS\data_save\D6/73.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    mss = msrt['ms']
    mz = msrt['mz']
    segno = msrt['segno']
    results = data['results']

    inds = [75]
    tzs = results[0]['tzs'][inds,:]
    RG = np.arange(np.min(tzs), np.max(tzs))
    dd = np.zeros((len(RG), len(mz)))
    xx = ncr.mat(int(RG[0]), int(RG[-1]), 1)
    for j in inds:
        newd = np.zeros((len(RG), len(mz)))
        tz, seg, x, segx, maxpos, irr = findTZ(ncr, rts[j], np.array(mss[j], ndmin=2), pw, w, 'RM')
        c, vsel,pc = gridsearch(x, maxpos, 6, irr)
        chrom, area, heig, d = polyfitting(c, x, vsel, np.array(mss[j], ndmin=2))
        re = np.searchsorted(RG, [tz[0], tz[1]])
        newd[re[0]:re[1],:]= d
        dd = dd+newd


    xx1 = ncr.mat(int(RG[0])-8, int(RG[-1])+8, 1)
    seg = np.array([[0, 8], [xx1['d'].shape[0]-8, xx1['d'].shape[0]]])
    fit, bas = backremv(xx1['d'],seg)
    bbas = bas[8:xx1['d'].shape[0]-8,:]

    if bass == 'baseline':
        xxo = xx['d'] - bbas
        resn = xx['d'] - bbas - dd
    else:
        xxo = xx['d']
        resn = xx['d'] - dd


    un = np.sum((np.power(resn, 2)))
    sstn = np.sum(np.power(xxo, 2))
    R2 = (sstn-un)/sstn
    print(R2)

    plt.plot(np.sum(xx1['d'],axis=1),'b')
    # plt.plot(range(8,xx1['d'].shape[0]-8),np.sum(dd,axis=0),'r')
    t = range(8,xx1['d'].shape[0]-8)
    plt.plot(t,np.sum(dd,axis=1),'r')
    plt.plot(np.sum(bas,axis=1),'k')
    plt.show()

def get_feature():
    pkl_file = open('PD6--RM--')
    # pkl_file = open('standardn2')
    data = pickle.load(pkl_file)
    msrt = data['MSRT']
    fn = "F:\MARS\data_save\D6/73.CDF"
    # fn = "F:\MARS\data_save\STDAND/zhi50-5vs1.CDF"
    options = data['options']
    ncr = netcdf_reader(fn, bmmap=False)
    pw = options['pw']
    w = options['w']
    thre = options['thres']
    rts = msrt['rt']
    ms = msrt['ms']

    com =49
    tz, seg, x, segx, maxpos, irr = findTZ(ncr, rts[com], np.array(ms[com], ndmin=2), pw, w, thre)
    c, vsel, pc = gridsearch(x, maxpos, 6, irr)
    c0 = ittfa(x, maxpos, 1)
    c1 = ittfa(x, maxpos, 2)
    c2 = ittfa(x, maxpos, 3)
    c3 = ittfa(x, maxpos, 4)
    c4 = ittfa(x, maxpos, 5)
    c5 = ittfa(x, maxpos, 11)
    #
    # fig1 = plt.figure(321)
    plt.subplot(322)
    plt.plot(x, color='k', linewidth=1.5, alpha=0.35)
    plt.subplot(324)
    plt.plot(c0, 'k')
    # plt.plot(c2, 'y')
    # plt.plot(c3, 'c')
    # plt.plot(c4, 'k')
    plt.plot(c5, 'b')
    plt.plot(c1, 'r')

    chrom, area, heig, d = polyfitting(c, x, vsel, np.array(ms[com], ndmin=2))
    ax2 = plt.subplot(326)
    plt.plot(np.sum(x, axis=1))
    plt.plot(chrom, 'r')
    ax2.set_xlabel('scans')

    com =48
    tz, seg, x, segx, maxpos, irr = findTZ(ncr, rts[com], np.array(ms[com], ndmin=2), pw, w, thre)
    c, vsel, pc = gridsearch(x, maxpos, 6, irr)
    c0 = ittfa(x, maxpos, 1)
    c1 = ittfa(x, maxpos, 2)
    c2 = ittfa(x, maxpos, 3)
    c3 = ittfa(x, maxpos, 4)
    c4 = ittfa(x, maxpos, 5)
    c5 = ittfa(x, maxpos, 11)
    #
    # fig1 = plt.figure(321)
    plt.subplot(321)
    plt.plot(x, color='k', linewidth=1.5, alpha=0.35)
    plt.subplot(323)
    # plt.plot(c0, 'g')
    plt.plot(c1, 'k')
    # plt.plot(c2, 'y')
    # plt.plot(c3, 'c')
    plt.plot(c5, 'b')
    plt.plot(c4, 'r')

    chrom, area, heig, d = polyfitting(c, x, vsel, np.array(ms[com], ndmin=2))
    ax1 = plt.subplot(325)
    plt.plot(np.sum(x, axis=1))
    plt.plot(chrom, 'r')
    ax1.set_xlabel('scans')
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9, hspace=0.35)
    plt.show()


if __name__ == '__main__':
    # plot_simulate()
    # plot_RM()
    # plot_compara1()
    # plot_compara2()
    # plot_fig5()
    # get_R2('baseline')
    get_R2(None)
    # get_feature()
