__author__ = 'Administrator'
import numpy as np
from matplotlib.pyplot import show, plot, text
import pickle


def svdX(X):
    u, s, v = np.linalg.svd(X)
    T = np.dot(u,s)
    return u, T,

def FR(x, s, o, com):
    z = range(0, min([min(s), min(o)]))+range(max([max(s),max(o)]), x.shape[0])
    xs = x[s,:]
    xs[xs<0]=0
    xz = x[z,:]
    xo = x[o,:]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))
    u, s0, v = np.linalg.svd(xc)
    t = np.dot(u[:,0:com],np.diag(s0[0:com]))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), np.sum(mc, 1))
    u1, s1, v1 = np.linalg.svd(x)
    t1 = np.dot(u1[:, 0:com], np.diag(s1[0:com]))
    c = np.dot(t1, r)
    c1, ind = contrains(c, s, o)

    spec = x[s[ind],:]
    cc = c1/c1[s[ind]]
    res_x = np.dot(np.array(cc, ndmin=2).T, np.array(spec, ndmin=2))
    xx = x - res_x
    return cc, xx

def contrains(c, s, o):
    ind_s = np.argmax(np.abs(c[s]))
    if c[s][ind_s] < 0:
        c = -c

    if s[0]<o[0]:
        if c[s[-2]]<c[s[-1]]:
            ind1 = s[-1]
            ind2 = o[np.argmax(c[o])]
        else:
            ind1 = s[np.argmax(c[s])]
            ind2 = o[0]
    else:
        if c[s[1]] < c[s[0]]:
            ind1 = o[np.argmax(c[o])]
            ind2 = s[0]
        else:
            ind1 = o[-1]
            ind2 = s[np.argmax(c[s])]

    for i, indd in enumerate(np.arange(ind1, 0, -1)):
        if c[indd-1] >= c[indd]:
            c[0:indd] = 0
            break
        if c[indd-1] < 0:
            c[0:indd] = 0
            break

    for i, indd in enumerate(np.arange(ind2, len(c)-1, 1)):
        if c[indd+1] >= c[indd]:
            c[indd+1:len(c)] = 0
            break
        if c[indd+1] < 0:
            c[indd+1:len(c)] = 0
            break
    return c, ind_s


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
        print(j)
        sx = x[j:j+w,:]
        u, s, v = np.linalg.svd(np.dot(sx,sx.T))
        em[j, :] = np.sqrt(s[0:mpc]) #np.log10(np.sqrt(s[0:mpc])) #
    return L, em

def backremv(x, seg):
    mn = np.shape(x)
    bak2 = np.zeros(mn)
    for i in range(0, mn[1]):
        tiab = []
        reg = []
        for j in range(0, len(seg)):
            tt = range(seg[j][0],seg[j][1])
            tiab.extend(x[tt, i])
            reg.extend(np.arange(seg[j][0], seg[j][1]))
        rm = reg - np.mean(reg)
        tm = tiab - np.mean(tiab)
        b = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
        s = np.mean(tiab)-np.dot(np.mean(reg), b)
        b_est = s+b*np.arange(mn[0])
        bak2[:, i] = x[:, i]-b_est
    bak = x-bak2
    return bak2, bak

if __name__ == '__main__':

    pkl_file = open('HELP_m.pkl')
    # pkl_file = open('standardn2')
    data = pickle.load(pkl_file)
    x = data['x']
    s = data['so']
    o = data['z']

    com=3
    # # plot(x)
    # # show()
    # c, xx = FR(x, s, o, 3)

    import profile
    profile.run("FR(x, s, o, com)")
    # plot(c)
    # show()

    # filename ="E:\MARS\data_save\D6/73.CDF"
    # ncr = netcdf_reader(filename, bmmap=False)
    # tic = ncr.tic()
    #
    # m = ncr.mat(6110,6168, 1)
    # # plot(m['d'])
    # # show()
    # fit, bas = backremv(m['d'], [(0,10),(50,57)])
    # # plot(fit)
    # # show()
    #
    # s = range(10, 23)
    # z = range(23, 35)
    # c, xx = FR(fit, s, z, 3)
    # plot(c)
    # show()
    # plot(xx)
    # show()
    #
    # L, em = FSWFA(xx, 3, 2)
    # plot(L,em)
    # show()

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
