__author__ = 'Administrator'

import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui

from FileFinder import FileFinderWidget
from REFPlot import REFPlotWidget
from SEGSTable11 import SEGSTable
from MCRALSset import MCRALSsetWidget
from MCRALSMASS import MCRALSMASSPlotDlg
from MCRALSCHROM import MCRALSCHROMPlotDlg
from TICPlot1 import TICPlot
from MSRTTable import MSRTTableWidget
from MASSDlg import MASSPlotDlg
from CHROMDlg import CHROMPlotDlg


class MainWindowWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MainWindowWidget, self).__init__(parent)
        # self.main_widget = QtGui.QWidget(self)

        self.fileWidget = FileFinderWidget()
        # self.fileWidget.sizePolicy(set)
        self.tabWidget = QTabWidget()

        ManualWidget = QWidget()
        VisualWidget = QWidget()

        self.refplot = REFPlotWidget()
        self.segtable = SEGSTable()
        self.mcralsset = MCRALSsetWidget()
        self.mcralsmass = MCRALSMASSPlotDlg()
        self.mcralschrom = MCRALSCHROMPlotDlg()
        self.okButton = QPushButton("OK")

        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.okButton)
        vbox = QVBoxLayout()
        vbox.addWidget(self.mcralsset)
        vbox.addWidget(self.mcralschrom)
        gridlayout= QGridLayout()
        gridlayout.addWidget(self.refplot, 0, 0, 1, 3)
        gridlayout.addWidget(self.segtable, 1, 0)
        gridlayout.addLayout(vbox, 1, 1)
        gridlayout.addWidget(self.mcralsmass, 1, 2)
        gridlayout.addLayout(hbox, 2, 2)
        ManualWidget.setLayout(gridlayout)

        self.ticplot = TICPlot()
        self.msrttable = MSRTTableWidget()
        self.massdlg = MASSPlotDlg()
        self.chromdlg = CHROMPlotDlg()

        gridlayout1= QGridLayout()
        # gridlayout.setSpacing(5)
        gridlayout1.addWidget(self.ticplot, 0, 0, 1, 3)
        gridlayout1.addWidget(self.msrttable, 1, 0)
        gridlayout1.addWidget(self.massdlg, 1, 2)
        gridlayout1.addWidget(self.chromdlg, 1, 1)
        VisualWidget.setLayout(gridlayout1)

        self.tabWidget.addTab(ManualWidget, "Manual")
        self.tabWidget.addTab(VisualWidget, "Visual")

        mainlayout = QHBoxLayout()
        mainlayout.addWidget(self.fileWidget)
        mainlayout.addWidget(self.tabWidget)
        self.setLayout(mainlayout)

        # self.connect(self.ticplot, SIGNAL("MASS_SELECT"), self.msplot.update_ms)

        # self.connect(self.ticplot, SIGNAL("RANGE_SELECT"), self.range_select)
        #
        self.connect(self.fileWidget, SIGNAL("ref_plot"), self.refplot.add_tic)
        self.connect(self.fileWidget, SIGNAL("tic_plot"), self.ticplot.add_tic)
        self.connect(self.segtable, SIGNAL("delete_segs"), self.refplot.update_segments)
        self.connect(self.refplot, SIGNAL("delete_SELECT"), self.segtable.undoreply)
        self.connect(self.refplot, SIGNAL("range_SELECT"), self.segtable.add_segs)

        self.connect(self.segtable, SIGNAL("seg_no"), self.mcralsset.updata_index)
        self.connect(self.segtable, SIGNAL("updataseg_no"), self.mcralsset.updata)
        # self.connect(self.segtable, SIGNAL("updeleteseg_no"), self.mcralsset.updata)

        # self.emit(SIGNAL("MASS_SELECT"), self.ncr, self.ind_right_press)
        self.connect(self.refplot, SIGNAL("MASS_SELECT"), self.mcralsmass.addrawmass)

        # self.connect(self.segtable, SIGNAL("seg_tic"), self.mcralschrom.addrawdata)

        self.connect(self.mcralsset, SIGNAL("reolved_plot"), self.mcralschrom.addresolvedchrom)
        self.connect(self.mcralsset, SIGNAL("reolved_plot"), self.mcralsmass.addresolvedmass)

        self.connect(self.segtable, SIGNAL("select_row"), self.mcralsset.updata_index)
        # self.connect(self.ffwidget, SIGNAL("TIC_plot"), self.ticplot.update_tics)
        # self.connect(self.ffwidget, SIGNAL("TIC_plot"), self.update_ncr)

        # self.fileWidget.setFixedSize(365, 800)
        # self.fileWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setWindowTitle("MARS -- MS(mass spectrum)-Assisted Resolution of Signal")
        # self.resize(600, 800)

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MainWindowWidget()
    window.show()
    sys.exit(app.exec_())

    #     filename1='D:/NetCDF/20121127_1UGNINGMENGXIANGMAOCAO05.CDF'
    #     filename2='D:/NetCDF/20121127_1UGNINGMENGXIANGMAOCAO03.CDF'
    #     tic1=netcdf_tic(filename1)
    #     tic2=netcdf_tic(filename2)
    #     form = TICPlot()
    #     form.add_tic(tic1)
    #     form.add_tic(tic2)
    #     form.show()