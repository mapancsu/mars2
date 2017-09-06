__author__ = 'Administrator'

import sys

from PyQt4.QtGui import *
from PyQt4 import QtGui

from TICPlot1 import TICPlot
from rioginal.MSRTTable1 import MSRTTableWidget
from MASSDlg1 import MASSPlotDlg
from CHROMDlg import CHROMPlotDlg


class VISUALWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(VISUALWidget, self).__init__(parent)

        self.ticplot = TICPlot()
        self.msrttable = MSRTTableWidget()
        self.massdlg = MASSPlotDlg()
        self.chromdlg = CHROMPlotDlg()

        # self.ticplot.move(10, 10)
        # self.msrttable.move(500, 10)
        # self.massdlg.move(500, 210)
        # self.msrttable.move(500, 620)

        # palette = QPalette()
        # palette.setColor(QPalette.Background, QColor(192, 253, 123))
        # self.mcralsset.setPalette(palette)

        # vbox = QVBoxLayout()
        # vbox.addWidget(self.mcralsset)
        # vbox.addWidget(self.mcralschrom)
        gridlayout= QGridLayout()
        # gridlayout.setSpacing(5)
        gridlayout.addWidget(self.ticplot, 0, 0, 1, 3)
        gridlayout.addWidget(self.msrttable, 1, 0)
        gridlayout.addWidget(self.massdlg, 1, 1)
        gridlayout.addWidget(self.chromdlg, 1, 2)

        self.setLayout(gridlayout)
        self.resize(1200, 1000)

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = VISUALWidget()
    window.show()
    sys.exit(app.exec_())