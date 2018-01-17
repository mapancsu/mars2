__author__ = 'Administrator'


def findTZ(data, MSRT, w, thre):
    d = data['d']
    rto = data['rt']
    ms = MSRT['MS']
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