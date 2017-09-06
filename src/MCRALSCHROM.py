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

class MCRALSCHROMPlotDlg(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

    def create_canvas(self):
        self.fig = plt.figure()
        self.axes = plt.subplot(111)
        self.axes.set_xlabel("scans")
        self.axes.set_ylabel("Intensity")
        self.axes.set_title("Resolved Chromatograms")
        self.canvas = FigureCanvas(self.fig)
        self.redraw()

    def redraw(self):
        self.canvas.draw()
        self.update()

    def addrawdata(self, segtic, segrt):
        self.axes.plot(segrt, segtic, 'r--')
        self.axes.legend(("TIC",))
        self.redraw()

    def addresolvedchrom(self, mzrange, chroms, specs):
        self.axes.plot(chroms, lw=1, c='b', alpha=.7, picker=5)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MCRALSCHROMPlotDlg()
    window.show()
    sys.exit(app.exec_())

