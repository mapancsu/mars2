__author__ = 'Administrator'

from PyQt4 import QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import sys
from chemoMethods import mcr_als, pcarep
import SVDDlg
import PUREDWIDGET

class MCRALSsetWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MCRALSsetWidget, self).__init__(parent)

        SegLabel = QLabel("Seg no")
        self.SegnoBox = QSpinBox()
        SegLabel.setBuddy(self.SegnoBox)
        self.SegnoBox.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)

        methodsLabel1 = QLabel("Component EST:")
        self.methodsComboBox1 = QComboBox()
        self.methodsComboBox1.addItem("SVD")
        self.methodsComboBox1.addItem("SCM")
        methodsLabel1.setBuddy(self.methodsComboBox1)

        methodsLabel2  = QLabel("Initial EST:")
        self.methodsComboBox2 = QComboBox()
        self.methodsComboBox2.addItem("Pure")
        self.methodsComboBox2.addItem("EFA")
        methodsLabel2.setBuddy(self.methodsComboBox2)

        nonnegativeLabel1 = QLabel("Non-negative:")
        self.nonnegativeBox1 = QComboBox()
        self.nonnegativeBox1.addItem("set to 0")
        self.nonnegativeBox1.addItem("nnls")
        self.nonnegativeBox1.addItem("fnnls")
        self.nonnegativeBox1.setCurrentIndex(2)
        nonnegativeLabel1.setBuddy(self.nonnegativeBox1)

        unimodalityLabel = QLabel("Unimodality:")
        self.unimodalityBox = QComboBox()
        self.unimodalityBox.addItem("horizon")
        self.unimodalityBox.addItem("vetical")
        self.unimodalityBox.addItem("average")
        self.unimodalityBox.setCurrentIndex(2)
        unimodalityLabel.setBuddy(self.unimodalityBox)

        nonnegativeLabel2 = QLabel("Non-negative:")
        self.nonnegativeBox2 = QComboBox()
        self.nonnegativeBox2.addItem("set to 0")
        self.nonnegativeBox2.addItem("nnls")
        self.nonnegativeBox2.addItem("fnnls")
        self.nonnegativeBox2.setCurrentIndex(2)
        nonnegativeLabel2.setBuddy(self.nonnegativeBox2)

        Masslabel = QLabel("MASS:")
        ChromLabel = QLabel("CHROM:")
        self.startButton1 = QPushButton("Start")

        mainLayout = QGridLayout()
        mainLayout.addWidget(SegLabel, 0, 0)
        mainLayout.addWidget(self.SegnoBox, 0, 1)
        mainLayout.addWidget(methodsLabel1, 1, 0)
        mainLayout.addWidget(self.methodsComboBox1, 1, 1)
        mainLayout.addWidget(methodsLabel2, 1, 2)
        mainLayout.addWidget(self.methodsComboBox2, 1, 3)
        mainLayout.addWidget(ChromLabel, 2, 0, 1, 2)
        mainLayout.addWidget(Masslabel, 2, 2, 1, 2)
        mainLayout.addWidget(nonnegativeLabel1, 3, 0)
        mainLayout.addWidget(self.nonnegativeBox1, 3, 1)
        mainLayout.addWidget(nonnegativeLabel2, 3, 2)
        mainLayout.addWidget(self.nonnegativeBox2, 3, 3)
        mainLayout.addWidget(unimodalityLabel, 4, 0)
        mainLayout.addWidget(self.unimodalityBox, 4, 1)
        mainLayout.addWidget(self.startButton1, 5, 3)
        self.setLayout(mainLayout)

        self.startButton1.clicked.connect(self.pc_estimation)
        # self.startButton2.clicked.connect(self.initial)
        # self.startButton3.clicked.connect(self.mcrals)

    def pc_estimation(self):
        self.updata_x(self.SegnoBox.value()-1)
        if self.methodsComboBox1.currentText() == "SVD":
            svddlg = SVDDlg.SVDQDialg()
            svddlg.setModal(True)
            svddlg.updata_data(self.x)
            svddlg.move(QPoint(50, 50))
            svddlg.exec_()
            self.pc = svddlg.pc_numbers

        if self.pc == 1:
            u, s, v, x, sigma = pcarep(self.x['d'], self.pc)
            sumx = np.sum(x, 1)
            RESU = {}
            RESU['rt'] = self.seg[0] + np.argmax(sumx)
            RESU['mz'] = self.mzrange
            RESU['spec'] = np.array(x[np.argmax(sumx), :], ndmin=2)
            RESU['chro'] = np.array(sumx, ndmin=2)
            RESU['r2'] = sigma
            self.emit(SIGNAL("MSRT_list"), RESU)
            self.emit(SIGNAL("reolved_plot"), RESU)
        elif self.pc >= 2:
            # self.startButton2.setEnabled(True)
            # self.startButton3.setEnabled(True)
            if self.methodsComboBox2.currentText() == "Pure":
                puredlg = PUREDWIDGET.PUREQDialg()
                puredlg.setModal(True)
                puredlg.updata_data(self.x, self.pc)
                puredlg.move(QPoint(50, 50))
                puredlg.exec_()
                if np.any(puredlg.pures) == True:
                    self.mcrals(puredlg.pures)

    def mcrals(self, pure):
        options = {}
        options['mass'] = np.array([self.nonnegativeBox2.currentText(),
                                    self.unimodalityBox.currentText()])
        options['chrom'] = np.array([self.nonnegativeBox1.currentText(),
                                     self.unimodalityBox.currentText()])
        result = mcr_als(self.x, self.pc, pure, 50, options)
        RESU = {}
        RESU['rt'] = self.seg[0] + np.sort(np.argmax(result['copt'], 1))
        index = np.argsort(np.argmax(result['copt'], 1))
        RESU['mz'] = self.mzrange
        RESU['spec'] = result['sopt'][:, index]
        sums = np.zeros(result['copt'].shape)
        for ind in range(0, self.pc):
            sums[:, ind] = np.sum(np.dot(result['copt'][:, ind], result['sopt'][ind, :]), 1)
        RESU['chro'] = sums
        RESU['r2'] = result['r2opt']
        self.emit(SIGNAL("MSRT_list"), RESU)
        self.emit(SIGNAL("reolved_plot"), RESU)

    def updata_x(self, index):
        self.seg = self.segments[index]
        self.x = self.ncr.mat(self.seg[0], self.seg[1], 1)

    def updata(self, ncr, segments):
        self.ncr = ncr
        self.mzrange = np.linspace(ncr.mass_min, ncr.mass_max, num=ncr.mass_max - ncr.mass_min + 1)
        self.segments = segments
        self.SegnoBox.setRange(1, len(segments))
        self.SegnoBox.setValue(1)

    def updata_index(self, index):
        self.SegnoBox.setValue(index)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MCRALSsetWidget()
    window.show()
    sys.exit(app.exec_())