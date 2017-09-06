from __future__ import division
__author__ = 'Administrator'
'''
MARS (Massspectrum-Assited Resolution of Signal)

Created on 2015-11-10

@author: pandemar

'''
from scipy.linalg import norm
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import scipy
from NetCDF import netcdf_reader

# # pkl_file = open('heye32-3.pkl')
# pkl_file = open('C:\Users\Administrator\Desktop\li/MARS_Project1-1.pkl')
# data = pickle.load(pkl_file)
# # data['results'].pop(0)
# data['results'] = []
# # data['finish_files'] = data['files']['fn']
# data['finish_files'] = []
# output1 = open('C:\Users\Administrator\Desktop\li/MARS_Project1-1.pkl', 'wb')
# pickle.dump(data, output1)
# output1.close()

# msrt = data['MSRT']
# fn = "F:\BDY-Synchronize\pycharm-GUI\pycharm_project\storedata/32/12c.cdf"
# options = data['options']
# ncr = netcdf_reader(fn, bmmap=False)
# pw = options['pw']
# w = options['w']
# thre = options['thres']
# rts = msrt['rt']
# ms = msrt['ms']

pkl_file = open('heyenew')
data = pickle.load(pkl_file)
files = data['files']['files']
segs = [23.5, 24]
area = []
for fn in files:
    ncr = netcdf_reader(fn, bmmap=False)
    rts = ncr.tic()['rt']
    seg = np.searchsorted(rts, segs)
    xx = ncr.mat(seg[0], seg[1], 1)
    area.append(np.sum(xx['d']))
np.savetxt('stand', np.array(area, ndmin=2), delimiter=",", fmt="%s")