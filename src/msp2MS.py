import numpy as np
import re
import matplotlib.pyplot as plt

def msp2MS_eRah(filename, index):
    f = open(filename)
    lines = f.readlines()
    name_index = []
    numpeaks_index = []
    for i, line in enumerate(lines):
        line_c = format(line).lower()
        if "name" in line_c:
            name_index.append(i)
        if "num peaks" in line_c:
            numpeaks_index.append(i)
    rt_infor = lines[name_index[index-1]]
    print(rt_infor)

    mz = []
    val = []
    for j in range(numpeaks_index[index-1]+1, name_index[index]):
        aa = lines[j].strip("\n")
        if len(aa):
            a = re.split(';', aa)
            for i in range(0, len(a)):
                if a[i]!= ' ':
                    cc = len(a[i])
                    b = re.split(' ', a[i])
                    if '' in b:
                        b.remove('')
                    mz.append(float(b[0]))
                    val.append(float(b[1]))
    f.close()
    return rt_infor, mz, val

def msp2MS_mars(filename, index):
    f = open(filename)
    lines = f.readlines()
    name_index = []
    numpeaks_index = []
    for i, line in enumerate(lines):
        line_c = format(line).lower()
        if "name" in line_c:
            name_index.append(i)
        if "num peaks" in line_c:
            numpeaks_index.append(i)
    rt_infor = lines[name_index[index-1]]
    print(rt_infor)

    mz = []
    val = []
    for j in range(numpeaks_index[index-1]+1, name_index[index]):
        a = re.split(';', lines[j])
        bb = []
        for i, aa in enumerate(a):
            b = aa
            if "\n" in a[i]:
                b = aa.strip("\n")
            bb.append(b)

        for i in range(0, len(bb)):
            b = re.split('\t', bb[i])
            if '' in b:
                b.remove('')
            mz.append(float(b[0]))
            val.append(float(b[1]))
    f.close()

    return rt_infor, mz, val

def msp2MS_adap(filename, index):
    f = open(filename)
    lines = f.readlines()
    name_index = []
    numpeaks_index = []
    for i, line in enumerate(lines):
        line_c = format(line).lower()
        if "name" in line_c:
            name_index.append(i)
        if "num peaks" in line_c:
            numpeaks_index.append(i)
    rt_infor = lines[name_index[index-1]]
    print(rt_infor)

    mz = []
    val = []
    for j in range(numpeaks_index[index-1]+1, name_index[index]):
        aa = lines[j].strip("\n")
        if len(aa):
            b = re.split(' ',aa)
            mz.append(float(b[0]))
            val.append(float(b[1]))
    f.close()

    return rt_infor, mz, val


def msp2MS_amdis(filename, index):
    f = open(filename)
    lines = f.readlines()
    name_index = []
    numpeaks_index = []
    for i, line in enumerate(lines):
        line_c = format(line).lower()
        if "name" in line_c:
            name_index.append(i)
        if "num peaks" in line_c:
            numpeaks_index.append(i)

    rt_infor = lines[name_index[0]]
    print(rt_infor)

    mz = []
    val = []
    for j in range(numpeaks_index[index-1]+1, len(lines)):
        # print(j)
        a = re.split(';', lines[j])
        bb = []
        for i, aa in enumerate(a):
            b = aa
            if "\n" not in a[i]:
                bb.append(b)

        for i in range(0, len(bb)):
            b = re.split(' ', bb[i])
            # b = re.split(' ', bb[i])
            if '' in b:
                b.remove('')
            mz.append(float(b[0]))
            val.append(float(b[1]))
    f.close()

    return rt_infor, mz, val

def msp2MS_MSDIAL(filename, index):
    f = open(filename)
    lines = f.readlines()
    name_index = []
    numpeaks_index = []
    for i, line in enumerate(lines):
        line_c = format(line).lower()
        if "name" in line_c:
            name_index.append(i)
        if "num peaks" in line_c:
            numpeaks_index.append(i)
    rt_infor = lines[name_index[index]]
    print(rt_infor)

    mz = []
    val = []
    for j in range(numpeaks_index[index-1]+1, name_index[index]):
        aa = lines[j].strip("\n")
        if len(aa):
            b = re.split('\t',aa)
            mz.append(float(b[0]))
            val.append(float(b[1]))
    f.close()

    return rt_infor, mz, val

if __name__ == '__main__':

    # axes = plt.subplot(515)
    xr = [35, 300]
    yr = [0, 1.2]
    si = 12
    top = 15


    axes6 = plt.subplot(611)
    rt_infor, mz, val = msp2MS_amdis("C:\Users\mapan\Desktop\msp/l-lecucine.msp", 1)
    val = np.array(val) / max(val)
    axes6.vlines(mz, np.zeros(len(val)), np.array(val) / max(val), color = 'r')
    indx = np.argsort(-np.array(val))
    for i in range(0, top):
        axes6.text(mz[indx[i]], val[indx[i]], str(int(mz[indx[i]])), size=si, color='b')
    axes6.tick_params(axis='both', labelsize=si)
    axes6.set_xlim(xr[0], xr[1])
    axes6.set_ylim(yr[0], yr[1])

    axes2 = plt.subplot(612)
    rt_infor, mz, val = msp2MS_mars("C:\Users\mapan\Desktop\msp/mars.msp", 1)
    axes2.vlines(mz, np.zeros(len(val)), np.array(val) / max(val), color = 'r')
    indx = np.argsort(-np.array(val))
    for i in range(0, top):
        axes2.text(mz[indx[i]], val[indx[i]], str(int(mz[indx[i]])), size=si,color='b')
    axes2.tick_params(axis='both', labelsize=si)
    axes2.set_xlim(xr[0], xr[1])
    axes2.set_ylim(yr[0], yr[1])

    axes4 = plt.subplot(613)
    rt_infor, mz, val = msp2MS_amdis("C:\Users\mapan\Desktop\msp/AMDIS.msp", 0)
    val = np.array(val) / max(val)
    axes4.vlines(mz, np.zeros(len(val)), val, color = 'r')
    indx = np.argsort(-np.array(val))
    for i in range(0, 10):
        axes4.text(mz[indx[i]], val[indx[i]], str(int(mz[indx[i]])), size=si,color='b')
    axes4.tick_params(axis='both', labelsize=si)
    axes4.set_xlim(xr[0], xr[1])
    axes4.set_ylim(yr[0], yr[1])

    axes3 = plt.subplot(614)
    rt_infor, mz, val = msp2MS_adap("C:\Users\mapan\Desktop\msp/ADAP-GC.msp", 118)
    val = np.array(val) / max(val)
    axes3.vlines(mz, np.zeros(len(val)), np.array(val) / max(val), color = 'r')
    indx = np.argsort(-np.array(val))
    for i in range(0, top):
        axes3.text(mz[indx[i]], val[indx[i]], str(int(mz[indx[i]])), size=si,color='b')
    axes3.tick_params(axis='both', labelsize=si)
    axes3.set_xlim(xr[0], xr[1])
    axes3.set_ylim(yr[0], yr[1])

    axes1 = plt.subplot(615)
    rt_infor, mz, val = msp2MS_eRah("C:\Users\mapan\Desktop\msp/eRah.msp", 185)
    val = np.array(val) / max(val)
    axes1.vlines(mz, np.zeros(len(val)), np.array(val)/max(val), color = 'r')
    indx = np.argsort(-np.array(val))
    for i in range(0, top):
        axes1.text(mz[indx[i]], val[indx[i]], str(int(mz[indx[i]])), size=si,color='b')
    axes1.tick_params(axis='both', labelsize=si)
    axes1.set_xlim(xr[0], xr[1])
    axes1.set_ylim(yr[0], yr[1])

    axes5 = plt.subplot(616)
    rt_infor, mz, val = msp2MS_MSDIAL("C:\Users\mapan\Desktop\msp/MSDIAL.msp", 301)
    val = np.array(val) / max(val)
    axes5.vlines(mz, np.zeros(len(val)), np.array(val) / max(val), color = 'r')
    indx = np.argsort(-np.array(val))
    for i in range(0, top):
        axes5.text(mz[indx[i]], val[indx[i]], str(int(mz[indx[i]])), size=si, color='b')
    axes5.tick_params(axis='both', labelsize=si)
    axes5.set_xlim(xr[0], xr[1])
    axes5.set_ylim(yr[0], yr[1])


    plt.subplots_adjust(bottom=0.05, top=0.99, left=0.08, right=0.95)
    plt.show()
