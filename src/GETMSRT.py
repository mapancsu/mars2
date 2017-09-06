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
from chemoMethods import pcarep
from LSFDlg import LSFQDialg
from SVDDlg import SVDQDialg
from PICKDlg import PICKQDialg
from MCRALSDlg import MCRALSQDialg
from HELPDlg import HELPQDialg
from ITTFADlg import ITTFAQDialg

class getmsrt(QDialog):
    def __init__(self, parent=None):
        super(getmsrt, self).__init__(parent)

        self.lsfdlg = LSFQDialg()
        self.svddlg = SVDQDialg()
        self.pickdlg = PICKQDialg()
        self.ittfadlg = ITTFAQDialg()
        self.helpdlg = HELPQDialg()
        self.mcralsdlg = MCRALSQDialg()

        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(self.lsfdlg, "LSF")
        self.tabWidget.addTab(self.svddlg, "SVD")
        self.tabWidget.addTab(self.pickdlg, "PICK")
        self.tabWidget.addTab(self.ittfadlg, "ITTFA")
        self.tabWidget.addTab(self.helpdlg, "HELP")
        self.tabWidget.addTab(self.mcralsdlg, "MCRALS")

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # self.buttonBox.button(QDialogButtonBox.Ok).setDefault(True)
        # self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.connect(self.buttonBox, SIGNAL("rejected()"), self.close)
        self.connect(self.buttonBox, SIGNAL("accepted()"), self.accept)

        VerticalLayout = QVBoxLayout()
        VerticalLayout.addWidget(self.tabWidget)
        VerticalLayout.addWidget(self.buttonBox)

        self.setLayout(VerticalLayout)
        self.resize(1000, 600)
        self.move(320, 75)

        self.connect(self.lsfdlg, SIGNAL("after_baseline"), self.update_baseline)
        # self.connect(self.svddlg, SIGNAL("do_svd"), self.do_svd)
        self.pickdlg.btnadd.on_clicked(self.updata_pick)
        self.ittfadlg.Addbtn.clicked.connect(self.add_data)
        self.helpdlg.addbtn.clicked.connect(self.add_helpdata)
        self.mcralsdlg.DoButton.clicked.connect(self.initial)
        self.mcralsdlg.startbutton.clicked.connect(self.start_mcr)

    def updata_pick(self,event):
        self.pickdlg.updata_data(self.x, int(self.svddlg.numberLineEdit.text()))

    def add_data(self):
        pc = int(self.svddlg.numberLineEdit.text())
        self.ittfadlg.add_data(self.x, pc)

    def add_helpdata(self):
        pc = int(self.svddlg.numberLineEdit.text())
        self.helpdlg.add_data(self.x, pc)

    def update_baseline(self, x):
        self.x = {'d':x, 'rt': self.rt, 'mz':self.mz}
        self.svddlg.updata_data(self.x)
        # self.pickdlg.updata_data(self.x, self.svddlg.pc_numbers)
        # self.helpdlg.updata_data(self.x, self.svddlg.pc_numbers)
        # self.mcralsdlg.updata_data(self.x, self.svddlg.pc_numbers)

    def initial(self):
        pc = int(self.svddlg.numberLineEdit.text())
        if pc != 0:
            # X = {'d':self.x, 'rt': self.rt, 'mz':self.mz}
            self.mcralsdlg.updata_data(self.x, pc)
            self.mcralsdlg.puremethod()
        else:
            msgBox = QMessageBox()
            msgBox.setText('please give pc numbers')
            msgBox.exec_()

    def start_mcr(self):
        if self.mcralsdlg.init == 1:
            self.mcralsdlg.mcrals()
        else:
            msgBox = QMessageBox()
            msgBox.setText('please give initial estimation')
            msgBox.exec_()

    def updata_data(self, X):
        self.x = X
        x = X['d']
        self.rt = X['rt']
        self.mz = X['mz']
        self.lsfdlg.updata_data(X)
        self.svddlg.updata_data(X)
        # self.pickdlg.updata_data(X)
        # self.helpdlg.updata_data(X)
        # self.mcralsdlg.updata_data(X)

    def accept(self):
        if self.tabWidget.currentIndex() == 2:
            self.RESU = self.pickdlg.get_resu()
        elif self.tabWidget.currentIndex() == 3:
            self.RESU = self.ittfadlg.get_resu()
        elif self.tabWidget.currentIndex() == 4:
            self.RESU = self.helpdlg.get_resu()
        elif self.tabWidget.currentIndex() == 5:
            self.RESU = self.mcralsdlg.get_resu()
        else:
            msgBox = QMessageBox()
            msgBox.setText("Switch GUI to PICK, HELP or MCRALS")
            msgBox.exec_()
            return
        if len(self.RESU):
            self.close()
        else:
            msgBox = QMessageBox()
            msgBox.setText("please get MSRT first")
            msgBox.exec_()

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = getmsrt()
    window.show()
    sys.exit(app.exec_())

