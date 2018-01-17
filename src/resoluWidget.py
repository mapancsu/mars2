__author__ = 'Administrator'

from PyQt4 import QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import sys
from GETMSRT import getmsrt

class ResoWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ResoWidget, self).__init__(parent)

        SegLabel = QLabel("SNO:")
        self.SegnoBox = QSpinBox()
        SegLabel.setBuddy(self.SegnoBox)
        self.SegnoBox.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
        self.startButton = QPushButton("Start")

        hbox = QHBoxLayout()
        hbox.addWidget(SegLabel)
        hbox.addWidget(self.SegnoBox)

        mainLayout = QVBoxLayout()
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
            if len(RESU):
                self.emit(SIGNAL("results"), RESU, self.seg, self.index)

    def updata_x(self, index):
        self.index = index
        rt = self.ncr.tic()['rt']
        seg = self.segments[index]
        self.seg = np.searchsorted(rt, seg)
        self.x = self.ncr.mat(self.seg[0], self.seg[1], 1)

    def updata(self, segments):
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