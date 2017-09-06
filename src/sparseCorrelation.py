import numpy as np
from scipy import sparse

def corrcoef_ms(mz1,ms1,mz2,ms2):
    I1 = np.zeros(len(mz1))
    I2 = np.ones(len(mz2))
    J1 = mz1.astype(int)
    J2 = mz2.astype(int)
    I  = np.hstack((I1,I2))
    J  = np.hstack((J1,J2))
    V  = np.hstack((ms1,ms2))
    x = sparse.coo_matrix((V,(J,I)),shape=(max(J)+1,2)).tocsr()
    meanx = x.sum(axis=0)/float(x.shape[0])
    x=x-meanx
    covx=np.sqrt(np.array(x.T*x))
    return covx[0][1]*covx[0][1]/(covx[0][0]*covx[1][1])


if __name__ == "__main__":

    
    mz1 = np.array([1,2,3,4,5,100,111])
    mz2 = np.array([1,2,3,4,5,9,11])
    ms1 = np.array([4,5,7,8,3,1,2])
    ms2 = np.array([4,5,7,9,3,1,2])
    print corrcoef_ms(mz1,ms1,mz2,ms2)
