__author__ = 'Administrator'

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector, Button
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
from NetCDF import netcdf_reader

class MCRALSMASSPlotDlg(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.showing = 0
        self.curr_no = -1
        self.create_canvas()

        # self.chromQwidget = CHROMQWidget()

        self.tablewidget = QTabWidget()
        self.tablewidget.addTab(self.canvas,"MASS")
        # self.tablewidget.addTab(self.chromQwidget.canvas,"CHROM")

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(self.tablewidget)
        self.setLayout(mainlayout)

    def create_canvas(self):

        self.fig = plt.figure()
        self.axes1 = plt.subplot(211)
        self.axes2 = plt.subplot(212)
        self.axes1.tick_params(axis='both', labelsize=8)
        self.axes2.tick_params(axis='both', labelsize=8)

        # plt.subplots_adjust(left=0.06, right=0.95, bottom=0.08, top=0.95, wspace=0.25, hspace=0.25)
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.35)
        self.canvas = FigureCanvas(self.fig)

        self.axes1.set_title("Raw MASS:", fontsize=8.5)
        self.axes2.set_title("Resolved MASS", fontsize=8.5)
        self.axes2.set_xlabel("m/z")

        axnext = plt.axes([0.2, 0.01, 0.1, 0.075])
        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        self.btprev = Button(axnext, 'PREV')
        self.btnext = Button(axprev, 'NEXT')

        self.btprev.on_clicked(self.updata_pmass)
        self.btnext.on_clicked(self.updata_nmass)

        self.redraw()
        ymino1, ymaxo1 = self.axes1.get_ylim()
        xmino1, xmaxo1 = self.axes1.get_xlim()
        self.oxy1 = [(xmino1, xmaxo1), (ymino1, ymaxo1)]
        ymino2, ymaxo2 = self.axes2.get_ylim()
        xmino2, xmaxo2 = self.axes2.get_xlim()
        self.oxy2 = [(xmino2, xmaxo2), (ymino2, ymaxo2)]
        self.update_axisrange(self.oxy1, self.oxy2)

    def redraw(self):
        self.canvas.draw()
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
        self.axes1.clear()
        self.ms=mass
        self.x=mass['mz']
        self.y=100*mass['val']/np.max(mass['val'])
        self.axes1.vlines(self.x, np.zeros((len(self.x),)), self.y, color='k', linestyles='solid')
        # self.axes1.xaxis.set_major_formatter(FormatStrFormatter('%3.4f'))
        self.oxy1[0] = (int(min(self.x)), math.ceil(max(self.x)))
        self.update_axisrange(self.oxy1, self.oxy2)
        self.axes1.set_ylim(min(self.y), max(self.y)*1.1)
        self.axes1.set_title("Raw MASS: rt="+str(np.round(rt, 3)), fontsize=8.5)
        self.canvas.draw()

    def add_resolmass(self, RESU, seg, no):
        self.RESU = RESU
        self.sums = RESU['pc']
        self.no = no
        mass = self.RESU['ms'][0]
        self.rang = self.RESU['mz']
        self.titlsub = "Seg("+str(no+1)+"),"+str(np.round(seg[0],3))+\
                       "~"+str(np.round(seg[-1],3))+"min,"
        self.add_ms(1, self.sums, self.rang, mass, 'b')
        # self.add_resolchrom()
        # self.prevbutton.setEnabled(True)
        # self.nextbutton.setEnabled(True)
        # self.chrobutton.setEnabled(True)

    # def add_resolchrom(self):
    #     # self.addrawmass(self.RESU, self.seg, self.no)
    #     self.chromQwidget.updata_data(self.RESU['tic'], self.RESU['chro'], self.titlsub)
    #     # self.chromQwidget.show()
    #     self.showing = 1

    # def clear_plot(self):
    #     if self.showing == 1:
    #         self.chromQwidget.close()
    #         self.showing = 0
    #     self.axes2.clear()
    #     self.axes2.set_xlabel("m/z")
    #     self.axes2.set_title("Resolved Mass:", fontsize=8.5)
        # self.prevbutton.setEnabled(False)
        # self.nextbutton.setEnabled(False)
        # self.chrobutton.setEnabled(False)

    def updata_pmass(self, event):
        if self.curr_no >= 2:
            self.curr_no = self.curr_no-1
            mass = self.RESU['ms'][self.curr_no-1]
            self.add_ms(self.curr_no, self.sums, self.rang, mass, 'b')

    def updata_nmass(self, event):
        if self.curr_no>=0:
            if self.curr_no < self.sums:
                self.curr_no = self.curr_no+1
                mass = self.RESU['ms'][self.curr_no-1]
                self.add_ms(self.curr_no, self.sums, self.rang, mass, 'b')

    def add_ms(self, no, sums, rang, mass, col):
        self.axes2.clear()
        self.axes2.set_title("Resolved mass: "+self.titlsub+"("+str(no)+"/"+str(sums)+")", fontsize=8.5)
        self.axes2.set_xlabel("m/z")
        self.axes2.vlines(rang, np.zeros((len(rang),)), mass, col, linestyles='solid')
        # self.axes2.xaxis.set_major_formatter(FormatStrFormatter('%3.4f'))
        vsel = np.nonzero(mass >= 0.001*np.max(mass))[0]
        self.oxy2[0] = (min(rang[vsel]), max(rang[vsel]))
        self.update_axisrange(self.oxy1, self.oxy2)
        t1 = [np.min(mass),np.max(mass)*1.1]
        self.axes2.set_ylim(np.min(mass), np.max(mass)*1.1)
        self.curr_no = no
        self.redraw()

    def clear_data(self):
        self.axes1.clear()
        self.axes1.set_title("Raw MASS:", fontsize=8.5)
        self.clear_plot()
        self.redraw()

    def loading(self):
        self.axes1.clear()
        self.axes2.clear()
        del self.axes1.collections[:]
        del self.axes2.collections[:]
        self.axes1.set_title("Raw MASS", fontsize=8.5)
        self.axes2.set_title("Resolved MASS",fontsize=8.5)
        self.axes2.set_xlabel("m/z")
        #self.chromQwidget.loading()

class CHROMQWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(self.canvas)
        self.setLayout(mainlayout)

    def create_canvas(self):
        self.fig = plt.figure()
        self.axes = plt.subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.axes.set_xlabel("scans")
        self.axes.set_ylabel("intensity")
        self.axes.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.35)
        self.canvas.draw()

    def redraw(self):
        self.canvas.draw()
        self.update()

    def updata_data(self, tic, chroms, subtitle):
        self.axes.clear()
        self.axes.plot(chroms.T, 'go--', label='resolved')
        self.axes.plot(tic, 'rs-', label='origin')
        self.axes.set_title("Resolved Chrom: "+subtitle, fontsize=8.5)
        # self.axes.legend(handles=[line1, line2], locals=3)
        # self.fig.legend((line1, line2), ('resolved', 'origin'))
        # plt.legend(bbox_to_anchor=(0.8, 1.0), loc=2, borderaxespad=0.)
        self.redraw()
        # plt.legend(handles=[line1, line2])

    def loading(self):
        self.axes.clear()
        del self.axes.collections[:]
        self.axes.set_xlabel("scans")
        self.axes.set_ylabel("intensity")

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MCRALSMASSPlotDlg()
    window.show()
    sys.exit(app.exec_())

