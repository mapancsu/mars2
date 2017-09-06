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

class CHROMPlotDlg(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()
        self.nextButton = QPushButton("Next")
        self.previousButton = QPushButton("Previous")
        self.addnextButton = QPushButton("Add next")

        # filename = 'E:/pycharm_project/n21.cdf'
        # self.ncrs = netcdf_reader(filename, bmmap=True)
        # self.add_tic(self.ncrs.tic())

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.previousButton)
        hbox.addWidget(self.nextButton)
        hbox.addWidget(self.addnextButton)
        hbox.addStretch()
        vbox.addLayout(hbox)
        vbox.addWidget(self.canvas)

        self.setLayout(vbox)
        self.resize(500, 800)

    def create_canvas(self):

        self.fig = plt.figure()
        self.axes1 = plt.subplot(211)
        self.axes2 = plt.subplot(212)
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.35)

        # self.axes1.set_xlabel("rt(min)")
        self.axes1.set_ylabel("Intensity")
        self.axes2.set_xlabel("rt(min)")
        self.axes2.set_ylabel("Intensity")
        #plt.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.fig)

        self.axes1.set_title("Resolved Chromatogram")
        self.axes2.set_title("Resolved Chromatograms")
        self.redraw()

    def redraw(self):
        self.canvas.draw()
        self.update()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = CHROMPlotDlg()
    window.show()
    sys.exit(app.exec_())
