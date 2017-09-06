__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
from scipy.linalg import norm
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from chemoMethods import mcr_als, pcarep, pure

class ITERWIDGET(QWidget):
    def __init__(self, parent=None):
        super(ITERWIDGET, self).__init__(parent)

        self.create_canvas()

        IterLabel = QLabel("Iteration")

        optiterLabel = QLabel("itopt:")
        self.optiterQtext = QLineEdit()
        self.optiterQtext.setEnabled(False)

        lofpcaLaber = QLabel("lof_pca:")
        self.lofpcaQtext = QLineEdit()
        self.lofpcaQtext.setEnabled(False)

        lofexpLaber = QLabel("lof_exp:")
        self.lofexpQtext = QLineEdit()
        self.lofexpQtext.setEnabled(False)

        r2Laber = QLabel("r2:")
        self.r2Qtext = QLineEdit()
        self.r2Qtext.setEnabled(False)

        nonnegativeLabel1 = QLabel("Chrom NNeg:")
        self.nonnegativeBox1 = QComboBox()
        self.nonnegativeBox1.addItem("set to 0")
        self.nonnegativeBox1.addItem("nnls")
        self.nonnegativeBox1.addItem("fnnls")
        self.nonnegativeBox1.setCurrentIndex(2)
        nonnegativeLabel1.setBuddy(self.nonnegativeBox1)

        unimodalityLabel = QLabel("Chrom Unimod:")
        self.unimodalityBox = QComboBox()
        self.unimodalityBox.addItem("horizon")
        self.unimodalityBox.addItem("vetical")
        self.unimodalityBox.addItem("average")
        self.unimodalityBox.setCurrentIndex(2)
        unimodalityLabel.setBuddy(self.unimodalityBox)

        nonnegativeLabel2 = QLabel("Mass NNeg:")
        self.nonnegativeBox2 = QComboBox()
        self.nonnegativeBox2.addItem("set to 0")
        self.nonnegativeBox2.addItem("nnls")
        self.nonnegativeBox2.addItem("fnnls")
        self.nonnegativeBox2.setCurrentIndex(2)
        nonnegativeLabel2.setBuddy(self.nonnegativeBox2)

        self.startbutton = QPushButton("Start")

        gridbox = QGridLayout()
        gridbox.addWidget(optiterLabel, 0, 0)
        gridbox.addWidget(self.optiterQtext, 0, 1)
        gridbox.addWidget(lofpcaLaber, 1, 0)
        gridbox.addWidget(self.lofpcaQtext, 1, 1)
        gridbox.addWidget(lofexpLaber, 2, 0)
        gridbox.addWidget(self.lofexpQtext, 2, 1)
        gridbox.addWidget(r2Laber, 3, 0)
        gridbox.addWidget(self.r2Qtext, 3, 1)

        gridbox1 = QGridLayout()
        gridbox1.addWidget(nonnegativeLabel1, 0, 0)
        gridbox1.addWidget(self.nonnegativeBox1, 0, 1)
        gridbox1.addWidget(unimodalityLabel, 1, 0)
        gridbox1.addWidget(self.unimodalityBox, 1, 1)
        gridbox1.addWidget(nonnegativeLabel2, 2, 0)
        gridbox1.addWidget(self.nonnegativeBox2, 2, 1)
        gridbox1.addWidget(self.startbutton, 3, 1)

        vbox = QVBoxLayout()
        vbox.addLayout(gridbox)
        vbox.addStretch()
        vbox.addLayout(gridbox1)

        mainlayout = QGridLayout()
        mainlayout.addWidget(IterLabel, 0, 2)
        mainlayout.addWidget(self.canvas, 1, 0, 4, 3)
        mainlayout.addLayout(vbox, 1, 3)
        self.setLayout(mainlayout)
        # self.startbutton.clicked.connect(self.mcrals)

    def create_canvas(self):
        self.fig = plt.figure()
        self.axes = plt.subplot(111)
        # plt.subplots_adjust(left=0.05, right=0.1, bottom=0.05, top=0.1)
        self.canvas = FigureCanvas(self.fig)
        self.axes.set_title("Chrom plot:")
        self.axes.set_xlabel("scans")
        self.redraw()


    def redraw(self):
        self.canvas.draw()
        self.update()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = ITERWIDGET()
    window.show()
    sys.exit(app.exec_())