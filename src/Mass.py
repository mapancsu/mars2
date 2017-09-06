'''
Created on 2013-6-25

@author: humblercoder
'''

import numpy as np

def binMS(ms):
    mz=ms['mz']
    val=ms['val']
    mz_min=40
    mz_max= 500
    rg=np.linspace(mz_min-0.5, mz_max+0.5,num=mz_max-mz_min+2)
    r=int(mz_max-mz_min+1)
    mz_bin  = np.zeros((r,),dtype=int)
    val_bin = np.zeros((r,),dtype=int)
    inds=np.searchsorted(mz, rg)
    for i in range(0,r):
        mz_bin[i] = (rg[i]+rg[i+1])/2
        val_bin[i]= sum(val[inds[i]:inds[i+1]])
    return {'mz':mz_bin, 'val':val_bin}

def top10MS(ms):
    mz=ms['mz']
    val=ms['val']
    ms_scaled=1000*val/np.max(val)
    
    inds=np.argsort(val)[::-1]
    mz_top10  = mz[inds]
    val_top10 = ms_scaled[inds]
    
    top10=[]
    for i in range(min(len(inds),10)):
            peak_intensity_MS=(mz_top10[i],val_top10[i])
            top10.append(peak_intensity_MS)
    return top10

if __name__ == '__main__':
    pass