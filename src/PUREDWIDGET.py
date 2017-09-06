__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from NetCDF import netcdf_reader
import numpy as np
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from chemoMethods import pcarep, pure


class PUREQWIDGET(QWidget):
    def __init__(self, parent=None):
        super(PUREQWIDGET, self).__init__(parent)

        self.createVariabletable()
        self.create_canvas()
        pureLabel = QLabel("Pure Variable Selection")
        purelistLabel = QLabel("Purest variable")

        DirectonLabel = QLabel("Direction:")
        self.DComboBox = QComboBox()
        self.DComboBox.addItem("Concentration")
        self.DComboBox.addItem("Spectrum")
        # self.DComboBox.setEnabled(False)
        self.DComboBox.setCurrentIndex(1)
        # self.DComboBox.setText(str(1))
        # self.DLineEdit.setStyleSheet("color:red")
        DirectonLabel.setBuddy(self.DComboBox)

        NoiseLabel = QLabel("Noise level:")
        self.NLinedit = QLineEdit()
        self.NLinedit.setText(str(10))
        self.NLinedit.setStyleSheet("color:red")
        NoiseLabel.setBuddy(self.NLinedit)

        self.DoButton = QPushButton("Start")

        vbox = QVBoxLayout()
        vbox.addWidget(purelistLabel)
        vbox.addWidget(self.VariableTable)

        gridbox = QGridLayout()
        gridbox.addWidget(DirectonLabel, 0, 0)
        gridbox.addWidget(self.DComboBox, 0, 1)
        gridbox.addWidget(NoiseLabel, 1, 0)
        gridbox.addWidget(self.NLinedit, 1, 1)
        gridbox.addWidget(self.DoButton, 2, 1)

        vvbox = QVBoxLayout()
        vvbox.addLayout(vbox)
        vvbox.addStretch()
        vvbox.addLayout(gridbox)

        mainlayout = QGridLayout()
        mainlayout.addWidget(pureLabel, 0, 2)
        mainlayout.addWidget(self.canvas, 1, 0, 4, 3)
        mainlayout.addLayout(vvbox, 1, 3)

        self.setLayout(mainlayout)
        # self.DoButton.clicked.connect(self.puremethod)

    def createVariabletable(self):
        self.VariableTable = QtGui.QTableWidget(0, 1)
        self.VariableTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.VariableTable.horizontalHeader().hide()
        self.VariableTable.verticalHeader().hide()
        self.VariableTable.setShowGrid(False)

    def redraw(self):
        self.canvas.draw()
        self.update()

    def create_canvas(self):
        self.fig = plt.figure()
        self.axes = plt.subplot(111)
        self.axes.set_xlabel("scans")
        self.axes.set_title("Purest variable figure")
        self.canvas = FigureCanvas(self.fig)
        self.redraw()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = PUREQWIDGET()
    window.show()
    sys.exit(app.exec_())