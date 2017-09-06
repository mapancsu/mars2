__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from NetCDF import netcdf_reader
import numpy as np
from scipy.linalg import norm
import sys
from chemoMethods import mcr_als, pcarep, pure
import SVDDlg
import LSFDlg
import MCRALSDlg
from GETMSRT import getmsrt

class ResoWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ResoWidget, self).__init__(parent)

        SegLabel = QLabel("SNO:")
        self.SegnoBox = QSpinBox()
        SegLabel.setBuddy(self.SegnoBox)
        self.SegnoBox.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)

        # PreproLabel = QLabel("Bas RED:")
        # self.PreproComboBox = QComboBox()
        # self.PreproComboBox.addItem("None")
        # self.PreproComboBox.addItem("LSF")
        # self.PreproComboBox.setCurrentIndex(0)
        # PreproLabel.setBuddy(self.PreproComboBox)
        #
        # ComESTLabel = QLabel("Com EST:")
        # self.ComESTComboBox = QComboBox()
        # self.ComESTComboBox.addItem("SVD")
        # self.ComESTComboBox.addItem("SCM")
        # self.ComESTComboBox.setCurrentIndex(0)
        # ComESTLabel.setBuddy(self.ComESTComboBox)
        #
        # ResMetLabel  = QLabel("Res Mth:")
        # self.ResMetComboBox = QComboBox()
        # self.ResMetComboBox.addItem("HELP")
        # self.ResMetComboBox.addItem("MCR-ALS")
        # self.ResMetComboBox.setCurrentIndex(1)
        # ResMetLabel.setBuddy(self.ResMetComboBox)

        self.startButton = QPushButton("Start")

        hbox = QHBoxLayout()
        hbox.addWidget(SegLabel)
        hbox.addWidget(self.SegnoBox)

        mainLayout = QVBoxLayout()
        # qtext = QLabel()
        # mainLayout = QGridLayout()
        # mainLayout.addWidget(SegLabel, 0, 0)
        # mainLayout.addWidget(self.SegnoBox, 0, 1)
        # mainLayout.addWidget(PreproLabel, 1, 0)
        # mainLayout.addWidget(self.PreproComboBox, 1, 1)
        # mainLayout.addWidget(ComESTLabel, 2, 0)
        # mainLayout.addWidget(self.ComESTComboBox, 2, 1)
        # mainLayout.addWidget(ResMetLabel, 3, 0)
        # mainLayout.addWidget(self.ResMetComboBox, 3, 1)
        # mainLayout.addWidget(qtext, 4, 0)
        mainLayout.addLayout(hbox)
        mainLayout.addWidget(self.startButton)
        mainLayout.addStretch()

        self.setLayout(mainLayout)

        self.startButton.clicked.connect(self.preprocess)

    def preprocess(self):
        if self.SegnoBox.value() != 0:
            self.updata_x(self.SegnoBox.value()-1)
            msrtdlg = getmsrt()
            msrtdlg.setModal(True)
            msrtdlg.updata_data(self.x)
            msrtdlg.exec_()
            RESU = msrtdlg.RESU
            self.emit(SIGNAL("results"), RESU, self.seg, self.index)

            # if self.PreproComboBox.currentText() == "LSF":
            #     lsfdlg = LSFDlg.LSFQDialg()
            #     lsfdlg.setModal(True)
            #     lsfdlg.updata_data(self.x)
            #     lsfdlg.exec_()
            #     self.x = lsfdlg.xx
            #
            # if self.ComESTComboBox.currentText() == "SVD":
            #     svddlg = SVDDlg.SVDQDialg()
            #     svddlg.setModal(True)
            #     svddlg.updata_data(self.x)
            #     svddlg.exec_()
            #     self.pc = svddlg.pc_numbers
            #
            # if self.pc == 1:
            #     u, s, v, x, sigma = pcarep(self.x['d'], self.pc)
            #     sumx = np.sum(x, axis=1)
            #     RESU = {}
            #     RESU['pc'] = self.pc
            #     # RESU['tic'] = np.sum(self.x['d'], axis=1)
            #     # RESU['rttic'] = self.x['rt']
            #     # RESU['rt'] = [self.x['rt'][np.argmax(sumx)]]
            #     RESU['spec'] = np.array(x[np.argmax(sumx), :]/norm(x[np.argmax(sumx), :]), ndmin=2)
            #     # RESU['chro'] = np.array(sumx, ndmin=2)
            #     RESU['r2'] = np.sum(np.power(x, 2))/np.sum(np.power(self.x['d'], 2))
            #     RESU['mz'] = self.x['mz']
            #     self.emit(SIGNAL("results"), RESU, self.seg, self.index)
            # elif self.pc >= 2:
            #     if self.ResMetComboBox.currentText() == "MCR-ALS":
            #         mcralsdlg = MCRALSDlg.MCRALSQDialg()
            #         mcralsdlg.setModal(True)
            #         mcralsdlg.updata_data(self.x, self.pc)
            #         mcralsdlg.show()
            #         QApplication.processEvents()
            #         mcralsdlg .exec_()
            #         RESU = mcralsdlg.RESU
            #         RESU['mz'] = self.x['mz']
            #         self.emit(SIGNAL("results"), RESU, self.seg, self.index)

    def updata_x(self, index):
        self.index = index
        rt = self.ncr.tic()['rt']
        seg = self.segments[index]
        self.seg = np.searchsorted(rt, seg)
        self.x = self.ncr.mat(self.seg[0], self.seg[1], 1)

    def updata(self, segments):
        # self.ncr = ncr
        # self.mzrange = np.linspace(ncr.mass_min, ncr.mass_max, num=ncr.mass_max - ncr.mass_min + 1)
        self.segments = segments
        if len(segments) >= 1:
            self.SegnoBox.setRange(1, len(segments))
            self.SegnoBox.setValue(1)
        elif len(segments) == 0:
            self.SegnoBox.setValue(0)

    def update_ncr(self, ncr):
        self.ncr = ncr

    def updata_index(self, index):
        self.SegnoBox.setValue(index)

    def loading(self, segments, ncr):
        self.update_ncr(ncr)
        self.updata(segments)

if __name__ == '__main__':

    # filename = 'E:/pycharm_project/others/t1(2).cdf'
    # ncr = netcdf_reader(filename, True)
    # tic = ncr.tic()
    # mz = ncr.mz_point(100)
    # # m = ncr.mat(1780, 1820, 1)
    # segments = [1780, 1820]
    app = QtGui.QApplication(sys.argv)
    window = ResoWidget()
    # window.updata(ncr, segments)
    window.show()
    sys.exit(app.exec_())