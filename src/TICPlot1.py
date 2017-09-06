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
from NetCDF import netcdf_reader

class TICPlot(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()
        self.segments = []

        # filename = 'E:/pycharm_project/n21.cdf'
        # self.ncrs = netcdf_reader(filename, bmmap=True)
        # self.add_tic(self.ncrs.tic())

    def create_canvas(self):
        self.fig = plt.figure()
        #self.fig.set_facecolor('white')
        self.canvas = FigureCanvas(self.fig)
        self.axes = plt.subplot(111)
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        plt.subplots_adjust(bottom=0.2, top=0.95)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addStretch()
        self.setLayout(vbox)

        self.zoom = RectangleSelector(self.axes, self.rectangle_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1], minspanx=5, minspany=5,
                                       spancoords='pixels')
        self.zoom_mode(True)
        self.redraw()
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]

    def redraw(self):
            self.canvas.draw()
            self.update()

    def zoom_mode(self, event):
        self.zoom.set_active(True)
        self.cidRelease = self.canvas.mpl_connect('button_release_event', self.mouse_release_callback)
        self.cidPress = self.canvas.mpl_connect('button_press_event', self.mouse_press_callback)
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw()
        print "zoom"

    def mouse_press_callback(self, event):
        if (event.button == 1 and event.dblclick == True):
            self.leftdblclick = True
        # if (event.button == 3 and event.dblclick == True):
        #     self.rightdblclick = True
        if event.button == 3:
            self.ind_right_press = np.searchsorted(self.x, event.xdata)
            del self.axes.collections[:]
            self.axes.vlines(event.xdata, self.oxy[1][0], self.oxy[1][1],
                                                  color='r', linestyles='--')
            self.redraw()
            self.emit(SIGNAL("MASS_SELECT"), self.ncr, self.ind_right_press)

    def mouse_release_callback(self, event):
        if (self.leftdblclick):
            self.leftdblclick = False
            del self.axes.collections[:]
            self.axes.set_xlim(self.oxy[0])
            self.axes.set_ylim(self.oxy[1])
            self.redraw()
        if (self.rightdblclick):
            self.rightdblclick = False
            del self.axes.collections[:]
            self.redraw()

    def rectangle_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if eclick.button == 1 and erelease.button == 1:
            self.axes.set_xlim(min(x1, x2), max(x1, x2))
            self.axes.set_ylim(min(y1, y2), max(y1, y2))
            self.redraw()

    def add_tic(self, ncr):
        tic = ncr.tic()
        self.ncr = ncr
        self.x = tic['rt']
        self.y = tic['val']
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.axes.plot(self.x, self.y, lw=1, c='b', alpha=.7, picker=5)
        self.axes.set_xlim(min(self.x), max(self.x))
        self.axes.set_ylim(min(self.y), max(self.y)*1.1)
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.redraw()

    def clear_data(self):
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.redraw()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = TICPlot()
    window.show()
    sys.exit(app.exec_())
