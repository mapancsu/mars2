'''
Created on 2013-6-25

@author: humblercoder
'''

import sqlite3
import numpy as np
from Mass import binMS,top10MS
from sparseCorrelation import corrcoef_ms
NIST_DBPath ='./Library/NIST2011.db'
MOL_DBPath ='./Library/NISTMol1.db'

class NISTSearch(object):

    def __init__(self,path):
        self.db = sqlite3.connect(NIST_DBPath)  
        self.cur = self.db.cursor()  
        self.cur.execute("select id,top10peakindex,top10peakintensity from catalog")
        c=self.cur.fetchall()
        self.top10NIST=[]
        for i in range(len(c)):
            id=c[i][0]
            mz=np.frombuffer(c[i][1],dtype=np.int)
            ms=np.frombuffer(c[i][2],dtype=np.int)
            self.top10NIST.append((id,mz,ms))

    def top10_screen(self,ms):
        self.ms_search=ms
        top10_search=top10MS(self.ms_search)
        self.top10_search=top10_search
        self.corrs_top10=[]
        for top10_lib in self.top10NIST:
            i=0
            for (x,y) in top10_search:
                if x in top10_lib[1]:
                    i=i+1
            corr=float(i)/float(len(top10_search))
            if corr > 0.4:
                self.corrs_top10.append((top10_lib[0],corr))
    
    def corr(self):
        self.corrs=[]
        for i in range(len(self.corrs_top10)):
            self.cur.execute("select peakindex,peakintensity  from catalog where id=%d"%(self.corrs_top10[i][0]))
            temp=self.cur.fetchall()
            masstemp=np.frombuffer(temp[0][0],dtype=np.int)
            intensitytemp=np.frombuffer(temp[0][1],dtype=np.int)
            temp2=(self.corrs_top10[i][0],corrcoef_ms(masstemp, intensitytemp,self.ms_search['mz'] , self.ms_search['val'] ))
            self.corrs.append(temp2)
        self.corrs.sort(key=lambda d:d[1],reverse=True)
    
if __name__ == "__main__":
    
    from NetCDF import netcdf_reader
    import time
    filename='E:\MARS\data_save\D6/73.CDF'
    ncr=netcdf_reader(filename)
    
    ms=binMS(ncr.mz_rt(20.50))
    ms_bg=binMS(ncr.mz_rt(20.60))

    ms['val']=np.abs(ms['val']-ms_bg['val'])  # background correction
    
    
    t = time.time()
    nist=NISTSearch(NIST_DBPath)
    nist.top10_screen(ms)
    nist.corr()
    print nist.corrs
    print time.time()-t

    