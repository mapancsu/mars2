__author__ = 'Administrator'


import sys
from PyQt4.QtGui import *
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class PICKQDialg(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()
        self.inds = []
        self.rt = []
        self.ms = []

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

        self.resize(800, 600)
        self.move(320, 75)
        self.setWindowTitle("PICK MSRT")

    def create_canvas(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = plt.subplot(111)
        self.axes.set_xlabel("Scans")
        self.axes.set_ylabel("Instensity")
        self.axes.set_title("PICK MSRT", fontsize=9)
        self.axes.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.22, top=0.90, left=0.08, right=0.9)

        axadd = plt.axes([0.09, 0.04, 0.08, 0.075])
        axundo = plt.axes([0.2, 0.04, 0.08, 0.075])
        self.btnadd = Button(axadd, 'add')
        self.btnundo = Button(axundo, 'undo')
        self.btnundo.on_clicked(self.undo_mode)

        self.redraw()
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]

        self.canvas.mpl_connect('button_press_event', self.mouse_press_callback)

    def redraw(self):
        self.canvas.draw()
        self.update()

    def undo_mode(self, event):
        if len(self.inds) >= 1:
            del self.axes.collections[:]
            self.inds.pop(-1)
            for ind in self.inds:
                self.axes.vlines(ind, self.oxy[1][0], self.oxy[1][1],
                                 color='g', linestyles='-')
            self.redraw()
        print "undo"

    def mouse_press_callback(self, event):
        if event.button == 3:
            self.xdata = event.xdata
            ind = np.searchsorted(range(int(self.oxy[0][0]), int(self.oxy[0][1])), event.xdata)
            self.inds.append(ind)
            rt = self.x['rt'][ind]
            ms = self.x['d'][ind, :]/np.linalg.norm(self.x['d'][ind, :])
            self.rt.append(rt)
            self.ms.append(ms)
            self.axes.vlines(event.xdata, self.oxy[1][0], self.oxy[1][1],
                             color='g', linestyles='-')
            self.redraw()

    def updata_data(self, x, pc):
        self.pc_numbers = pc
        self.axes.clear()
        self.x = x
        self.y = np.sum(x['d'], axis=1)
        self.axes.plot(self.x['d'], lw=1, alpha=.7, picker=5)
        self.axes.set_xlim(0, len(self.y))
        self.axes.set_ylim(0, np.max(self.x['d']) * 1.1)
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.redraw()

    def get_resu(self):
        if len(self.rt):
            RESU = {"methods": "P", "ms": self.ms, 'rt': self.rt, 'mz': self.x['mz'], 'pc':self.pc_numbers, 'R2': 'none'}
        else:
            RESU = []
        return RESU

if __name__ == '__main__':

    # filename = 'E:/pycharm_project/others/t1(2).cdf'
    # ncr = netcdf_reader(filename, True)
    # tic = ncr.tic()
    # mz = ncr.mz_point(100)
    # m = ncr.mat(1780, 1820, 1)
    app = QtGui.QApplication(sys.argv)
    window = PICKQDialg()
    # window.updata_data(m,3)
    window.show()
    sys.exit(app.exec_())