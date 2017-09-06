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


from REFPlot import REFPlotWidget
from SEGSTable11 import SEGSTable
from MCRALSset import MCRALSsetWidget
from MCRALSMASS import MCRALSMASSPlotDlg
from MCRALSCHROM import MCRALSCHROMPlotDlg


class MANUALWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MANUALWidget, self).__init__(parent)

        # self.refplot()
        # self.segtable()
        # self.mcralsset()
        # self.mcralsmass()
        # self.mcralschrom()

        self.refplot = REFPlotWidget()
        self.segtable = SEGSTable()
        self.mcralsset = MCRALSsetWidget()
        self.mcralsmass = MCRALSMASSPlotDlg()
        self.mcralschrom = MCRALSCHROMPlotDlg()

        # palette = QPalette()
        # palette.setColor(QPalette.Background, QColor(192, 253, 123))
        # self.mcralsset.setPalette(palette)

        vbox = QVBoxLayout()
        vbox.addWidget(self.mcralsset)
        vbox.addWidget(self.mcralschrom)
        gridlayout= QGridLayout()
        gridlayout.addWidget(self.refplot, 0, 0, 1, 3)
        gridlayout.addWidget(self.segtable, 1, 0)
        gridlayout.addLayout(vbox, 1, 1)
        gridlayout.addWidget(self.mcralsmass, 1, 2)

        self.setLayout(gridlayout)
        self.resize(1200, 1000)


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MANUALWidget()
    window.show()
    sys.exit(app.exec_())