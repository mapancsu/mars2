__author__ = 'Administrator'

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.widgets import RectangleSelector, SpanSelector, Button
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
from NetCDF import netcdf_reader

class MASSPlotDlg(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()
        self.create_chromcanvas()
        ymino1, ymaxo1 = self.axes1.get_ylim()
        xmino1, xmaxo1 = self.axes1.get_xlim()
        self.oxy1 = [(xmino1, xmaxo1), (ymino1, ymaxo1)]
        ymino2, ymaxo2 = self.axes2.get_ylim()
        xmino2, xmaxo2 = self.axes2.get_xlim()
        self.oxy2 = [(xmino2, xmaxo2), (ymino2, ymaxo2)]
        self.update_axisrange(self.oxy1, self.oxy2)

        self.fn = str()
        self.massplot = str()
        # self.nextButton = QPushButton("Next")
        # self.previousButton = QPushButton("Previous")
        # filename = 'E:/pycharm_project/n21.cdf'
        # self.ncrs = netcdf_reader(filename, bmmap=True)
        # self.add_tic(self.ncrs.tic())

        self.TABLEWIDGET = QTabWidget()
        self.TABLEWIDGET.addTab(self.canvas1, "MASS")
        self.TABLEWIDGET.addTab(self.canvas2, "CHROM")

        vbox = QHBoxLayout()
        vbox.addWidget(self.TABLEWIDGET)
        self.setLayout(vbox)

    def create_canvas(self):
        self.fig1 = plt.figure()
        self.axes1 = plt.subplot(211)
        self.axes2 = plt.subplot(212)
        self.axes1.tick_params(axis='both', labelsize=8)
        self.axes2.tick_params(axis='both', labelsize=8)
        # self.axes3 = plt.subplot(313)
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.35)
        self.canvas1 = FigureCanvas(self.fig1)
        self.axes1.set_title("Raw MASS",fontsize=8.5)
        self.axes2.set_title("Resolved MASS",fontsize=8.5)
        self.axes2.set_xlabel("m/z")

        axnext = plt.axes([0.2, 0.01, 0.1, 0.075])
        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        self.btprev = Button(axnext, 'PREV')
        self.btnext = Button(axprev, 'NEXT')

        self.redraw1()

    def create_chromcanvas(self):
        self.fig2 = plt.figure()
        self.axes3 = plt.subplot(111)
        self.axes3.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.35)
        self.canvas2 = FigureCanvas(self.fig2)
        self.axes3.set_title("CHROM", fontsize=8.5)
        self.axes3.set_xlabel("scan")
        self.axes3.set_ylabel("intensity")
        axnext = plt.axes([0.2, 0.01, 0.1, 0.075])
        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        self.btprev = Button(axnext, 'PREV')
        self.btnext = Button(axprev, 'NEXT')
        self.redraw2()

    def redraw1(self):
        self.canvas1.draw()
        self.update()

    def redraw2(self):
        self.canvas2.draw()
        self.update()

    def update_axisrange(self, oxy1, oxy2):
        if oxy1[0][0] == 0 or oxy2[0][0] == 0:
            xmin = max(oxy1[0][0], oxy2[0][0])
        else:
            xmin = min(oxy1[0][0], oxy2[0][0])
        xmax = max(oxy1[0][1], oxy2[0][1])
        self.axes1.set_xlim((xmin, xmax))
        self.axes2.set_xlim((xmin, xmax))

    def addrawmass(self, ncr, inds):
        mass = ncr.mz_point(inds)
        rts = ncr.scan_acquisition_time
        rt = rts[inds]/60
        # self.add_ms(0, mass, col='k')
        del self.axes1.collections[:]
        self.axes1.clear()
        # self.axes1.set_xlabel("m/z")
        self.axes1.set_ylabel("Instensity")
        self.ms = mass
        self.x = mass['mz']
        self.y = 100*mass['val']/np.max(mass['val'])
        self.axes1.vlines(self.x, np.zeros((len(self.x),)), self.y, color='k', linestyles='solid')
        # self.axes1.xaxis.set_major_formatter(FormatStrFormatter('%3.4f'))
        self.oxy1[0] = (int(min(self.x)), math.ceil(max(self.x)))
        self.update_axisrange(self.oxy1, self.oxy2)
        self.axes1.set_ylim(min(self.y), 1.1*max(self.y))
        self.axes1.set_title("Raw MASS: rt="+str(np.round(rt, 3)),fontsize=8.5)
        self.canvas1.draw()

    def addresolmass(self, rt, ms, mz, no, chrom, fn):
        self.axes2.clear()
        self.axes2.set_xlabel("m/z")
        self.axes2.set_ylabel("Instensity")
        y = 100*ms/np.max(ms)
        self.axes2.vlines(mz, np.zeros((len(mz),)), y, color='k', linestyles='solid')
        # self.axes1.xaxis.set_major_formatter(FormatStrFormatter('%3.4f'))
        vsel = np.nonzero(y >= 0.001*max(y))
        mzrange = np.array(mz)
        # tt = (min(mz[vsel]), max(mz[vsel]))
        self.oxy2[0] = (min(mzrange[vsel]), max(mzrange[vsel]))
        self.update_axisrange(self.oxy1, self.oxy2)
        # self.axes2.set_xlim(int(min(mz)), math.ceil(max(mz)))
        self.axes2.set_ylim(min(y), 1.2*max(y))
        self.axes2.set_title("MSRT=" + str(np.round(rt, 3)) + ',seg=' + str(no+1), fontsize=8.5)
        self.redraw1()
        self.massplot = str(np.round(rt, 3))

        self.axes3.clear()
        del self.axes3.collections[:]
        self.axes3.set_xlabel("Retention Time (min)")
        self.axes3.set_ylabel("Instensity")
        self.redraw2()
        if len(chrom) and len(fn):
            self.addchrom(chrom, fn)

    def addchrom(self, chrom, fn):
        rts = chrom['rts']
        chr = chrom['chrom']
        segs = chrom['segs']
        orc = chrom['orc']
        self.axes3.plot(segs, orc, 'rs-', label='origin')
        self.axes3.set_xlim(min(segs), max(segs))
        self.axes3.set_ylim(min(orc), 1.2 * max(orc))
        # self.axes3.xaxis.set_major_locator(MultipleLocator(10000))
        if len(rts) > 0:
            self.axes3.plot(rts, chr, 'go--', label='resolved')
            self.axes3.set_ylim(min([min(chr), min(orc)]), 1.2 * max(orc))
            # self.axes3.set_xlim(int(min(rts)), int(max(rts)))
            # self.axes3.set_ylim(min(chr), 1.2*max(chr))

        self.axes3.set_title("Resolved Chrom: " + str(fn), fontsize=8.5)
        self.redraw2()

    def updatamaxc(self, rows):
        self.maxc = rows

    def prevp(self):
        if len(self.massplot):
            self.emit(SIGNAL("prevp1"),self.massplot)
            if len(self.fn_str):
                self.emit(SIGNAL("prevp2"),self.massplot)

    def nextp(self):
        if len(self.massplot)!= 0 and self.massplot<self.maxc:
                self.emit(SIGNAL("next1"), self.massplot)
                if len(self.fn_str):
                    self.emit(SIGNAL("next2"), self.massplot)

    def clear_data(self):
        self.axes1.clear()
        self.axes2.clear()
        self.axes3.clear()
        del self.axes1.collections[:]
        del self.axes2.collections[:]
        del self.axes3.collections[:]
        self.axes1.set_title("Raw MASS", fontsize=8.5)
        self.axes2.set_title("Resolved MASS",fontsize=8.5)
        self.axes2.set_xlabel("m/z")
        self.axes3.set_xlabel("Retention Time (min)")
        self.axes3.set_ylabel("Instensity")

    def loading(self):
        self.clear_data()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MASSPlotDlg()
    window.show()
    sys.exit(app.exec_())
