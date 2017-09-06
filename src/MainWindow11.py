__author__ = 'Administrator'

import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
import numpy as np

from FileFinder import FileFinderWidget
from REFPlot import REFPlotWidget
from SEGSTable import SEGSTable
from resoluWidget import ResoWidget
from MCRALSMASS import MCRALSMASSPlotDlg
from TICPlot1 import TICPlot
from MSRTTable import MSRTTableWidget
from MASSDlg import MASSPlotDlg

from MARS_methods import mars


class MainWindowWidget(QtGui.QWidget):
    # def __init__(self):
    #     super(MainWindowWidget, self).__init__()
    #     QtGui.QMainWindow.__init__(self)
    def __init__(self, parent=None):
        super(MainWindowWidget, self).__init__(parent)
        self.create_menu_toolbar()
        self.Setparameter = SetmarsQDialg()

        self.fileWidget = FileFinderWidget()
        self.fileWidget.setMaximumWidth(100)
        self.fileWidget.setMaximumWidth(250)
        self.tabWidget = QTabWidget()
        ManualWidget = QWidget()
        VisualWidget = QWidget()

        self.refplot = REFPlotWidget()
        self.refplot.setMinimumHeight(200)
        self.segtable = SEGSTable()
        self.resoluset = ResoWidget()
        self.mcralsmass = MCRALSMASSPlotDlg()
        self.okButton = QPushButton("Update MSRT")

        self.ticplot = TICPlot()
        # self.ticplot.setMinimumHeight(200)
        self.ticplot.setMaximumHeight(396)
        self.msrttable = MSRTTableWidget()
        self.msrttable.setMaximumWidth(300)
        self.massdlg = MASSPlotDlg()
        # self.chromdlg = CHROMPlotDlg()
        self.setmarsbutton = QPushButton('Set MARS')
        self.startmasrbutton = QPushButton('Start MARS')
        self.marschromns = QPushButton("Show Chroms")

        self.connect(self.fileWidget, SIGNAL("delete_current"), self.segtable.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.refplot.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.mcralsmass.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.msrttable.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.ticplot.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.massdlg.clear_data)
        self.connect(self.fileWidget, SIGNAL("ref_plot"), self.refplot.add_tic)
        self.connect(self.fileWidget, SIGNAL("tic_plot"), self.ticplot.add_tic)

        self.connect(self.segtable, SIGNAL("delete_segs"), self.refplot.update_segments)
        self.connect(self.segtable, SIGNAL("seg_no"), self.resoluset.updata_index)
        self.connect(self.segtable, SIGNAL("select_row"), self.resoluset.updata_index)
        self.connect(self.segtable, SIGNAL("updataseg_no"), self.resoluset.updata)
        self.connect(self.segtable, SIGNAL("mass_plot"), self.mcralsmass.add_resolmass)
        self.connect(self.segtable, SIGNAL("msrt_list"), self.msrttable.add_msrt)

        self.connect(self.refplot, SIGNAL("delete_SELECT"), self.segtable.undoreply)
        self.connect(self.refplot, SIGNAL("range_SELECT"), self.segtable.add_segs)
        self.connect(self.refplot, SIGNAL("MASS_SELECT"), self.mcralsmass.addrawmass)

        self.connect(self.ticplot, SIGNAL("MASS_SELECT"), self.massdlg.addrawmass)

        self.connect(self.resoluset, SIGNAL("results"), self.segtable.updata_resulist)
        self.connect(self.okButton, SIGNAL("clicked()"), self.segtable.msrtlist)
        self.connect(self.setmarsbutton, SIGNAL("clicked()"), self.setparameter)
        self.startmasrbutton.clicked.connect(self.getmars_results)
        # self.connect(self.segtable, SIGNAL("chro_plot"), self.mcralsmass.add_resolchrom)
        # self.connect(self.resoluset, SIGNAL("MSRT_list"), self.segtable.add_result)
        # self.connect(self.resoluset, SIGNAL("reolved_plot"), self.massdlg.add_data)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addStretch()
        vbox.addWidget(self.okButton)
        hbox.addStretch()
        hbox.addLayout(vbox)

        gridlayout= QGridLayout()
        gridlayout.addWidget(self.refplot, 0, 0, 1, 5)
        gridlayout.addWidget(self.segtable, 1, 0, 1, 3)
        gridlayout.addWidget(self.mcralsmass, 1, 3, 1, 2)
        gridlayout.addWidget(self.resoluset, 2, 0, 1, 3)
        gridlayout.addLayout(hbox, 2, 3)
        ManualWidget.setLayout(gridlayout)

        marshbox = QHBoxLayout()
        marshbox.addWidget(self.setmarsbutton)
        marshbox.addWidget(self.startmasrbutton)
        marshbox.addStretch()
        marshbox.addWidget(self.marschromns)

        gridlayout1= QGridLayout()
        # gridlayout.setSpacing(5)
        gridlayout1.addWidget(self.ticplot, 0, 0, 1, 5)
        gridlayout1.addWidget(self.msrttable, 1, 0, 1, 1)
        gridlayout1.addWidget(self.massdlg, 1, 1, 1, 4)
        # gridlayout1.addWidget(self.chromdlg, 1, 1)
        gridlayout1.addLayout(marshbox, 2, 0, 1, 5)
        VisualWidget.setLayout(gridlayout1)

        self.tabWidget.addTab(ManualWidget, "Manual")
        self.tabWidget.addTab(VisualWidget, "Visual")

        mainlayout = QHBoxLayout()
        mainlayout.addWidget(self.fileWidget)
        mainlayout.addWidget(self.tabWidget)
        self.setLayout(mainlayout)

        # self.resize(1200, 900)
        self.setMinimumWidth(1300)
        self.setMinimumHeight(900)
        self.setWindowTitle("MARS -- MS-Assisted Resolution of Signal")

    def setparameter(self):
        self.Setparameter.setModal(True)
        # self.Setparameter.conditions()
        self.options = self.Setparameter.conditions()
        self.Setparameter.move(QPoint(50, 50))
        self.Setparameter.exec_()

    def getmars_results(self):
        self.files = self.fileWidget.get_files()
        self.MSRT = self.msrttable.get_MSRT()
        self.options = self.Setparameter.conditions()
        self.results = []
        if len(self.MSRT) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please input MSRT.")
            msgBox.exec_()
        elif len(self.files) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please load files.")
            msgBox.exec_()
        else:
            ncrs = self.files['ncrs']
            for ncr in ncrs:
                result = mars(ncr, self.MSRT, self.options)
                self.results.append(result)
        self.getQUANQUAL_table()
        return self.files, self.MSRT, self.results

    def getQUANQUAL_table(self):
        if len(self.MSRT)>=1 and len(self.files) >=1:
            pre_vec = ['comID', 'RT.ref(min)']
            file_str = []
            for i in range(len(self.files['names'])):
                file_str.append(str(self.files['names'][i]))
            pre_vec.extend(file_str)
            fn_vec = np.array(pre_vec)

            area = np.zeros((len(self.files['names'])+2, len(self.MSRT['rt'])))
            area[0, :] = range(1, len(self.MSRT['rt'])+1)
            area[1, :] = self.MSRT['rt']
            for ind, result in enumerate(self.results):
                area[ind+2, :] = result['areas'].T
            ddtype = [('0', 'S20')]
            for i in range(1, len(self.MSRT['rt'])+1):
                ddtype.append((str(i), float))
            # ab = np.zeros(fn_vec.size, dtype=[('1', 'S6'), ('2', float)])
            # ab['var1'] = fn_vec
            # ab['var2'] = area.T
            ab = np.zeros(fn_vec.size, ddtype)
            ab['0'] = fn_vec
            for i in range(1, len(self.MSRT['rt'])+1):
                ab[str(i)] = area[:, i-1].T
            # np.savetxt('QUAN_tablemp.txt', ab, fmt="%20s")
            np.savetxt('QUAN_tablemp.txt', ab, delimiter=';', fmt="%10s")

    def create_menu_toolbar(self):
        saveOpenAction = QtGui.QAction(QtGui.QIcon('images/save.png'), '&Save', self)
        saveOpenAction.setShortcut('Ctrl+S')
        saveOpenAction.setStatusTip('Save results')

        saveasOpenAction = QtGui.QAction(QtGui.QIcon('images/saveas.png'), '&Save as', self)
        saveasOpenAction.setShortcut('Ctrl+A')
        saveasOpenAction.setStatusTip('Save results as')

        saveProjectionAction = QtGui.QAction(QtGui.QIcon('images/saveprojection.png'), '&Save projection', self)
        saveProjectionAction.setShortcut('Ctrl+P')
        saveProjectionAction.setStatusTip('Save projection')

        saveProjectionasAction = QtGui.QAction(QtGui.QIcon('images/saveprojectionas.png'), '&Save projection as', self)
        saveProjectionasAction.setShortcut('Ctrl+M')
        saveProjectionasAction.setStatusTip('Save projection as')

        OpenProjectionAction = QtGui.QAction(QtGui.QIcon('images/openprojection.png'), '&Open projection', self)
        OpenProjectionAction.setShortcut('Ctrl+O')
        OpenProjectionAction.setStatusTip('Open saved projection')

        ExitAction = QtGui.QAction(QtGui.QIcon('images/save.png'), '&Open projection', self)
        ExitAction.setShortcut('Ctrl+E')
        ExitAction.setStatusTip('Exit')
        ExitAction.connect(ExitAction, SIGNAL('triggered()'), QtGui.qApp, SLOT('quit()'))

        aboutme = QtGui.QAction('About me', self)
        updata = QtGui.QAction('Check for updata', self)


        # menubar = self.menuBar()
        # file = menubar.addMenu('&File')
        # file.addAction(saveOpenAction)
        # file.addAction(saveasOpenAction)
        # file.addAction(saveProjectionAction)
        # file.addAction(saveProjectionasAction)
        # file.addAction(OpenProjectionAction)
        # file.addAction(ExitAction)
        # # resolu = menubar.addMenu('&Resolution')
        # # resolu.addAction(selectreference)
        # # resolu.addAction(resolution)
        # # marsmain = menubar.addMenu('&MARS main')
        # # marsmain.addAction(setpara)
        # # marsmain.addAction(start)
        # # marsmain.addAction(stop)
        # helpme = menubar.addMenu('&Help')
        # helpme.addAction(aboutme)
        # helpme.addAction(updata)

class SetmarsQDialg(QDialog):
    def __init__(self, parent=None):
        super(SetmarsQDialg, self).__init__(parent)
        pwLabel = QLabel('PW')
        thresLabel = QLabel('Thres')
        windowLabel = QLabel('W')
        self.pwQtext = QLineEdit()
        self.pwQtext .setEnabled(True)
        self.pwQtext.setText(str(50))
        pwLabel.setBuddy(self.pwQtext)
        self.thresQspin = QDoubleSpinBox()
        self.thresQspin.setEnabled(True)
        self.thresQspin.setRange(0.90, 0.95)
        self.thresQspin.setSingleStep(0.01)
        # self.thresQspin.setSingleStep(0.01)
        self.thresQspin.setValue(0.90)
        thresLabel.setBuddy(self.thresQspin)
        self.windowQSpin = QSpinBox()
        self.windowQSpin.setEnabled(True)
        self.windowQSpin.setRange(3, 7)
        self.windowQSpin.setValue(5)
        windowLabel.setBuddy(self.windowQSpin)

        mainlayout = QGridLayout()
        mainlayout.addWidget(pwLabel, 0, 0)
        mainlayout.addWidget(self.pwQtext, 0, 1)
        mainlayout.addWidget(thresLabel, 1, 0)
        mainlayout.addWidget(self.thresQspin, 1, 1)
        mainlayout.addWidget(windowLabel, 2, 0)
        mainlayout.addWidget(self.windowQSpin, 2, 1)
        self.setLayout(mainlayout)

    def conditions(self):
        pw = int(self.pwQtext.text())
        thres = self.thresQspin.value()
        w = self.windowQSpin.value()
        options = {'pw': pw, 'thres': thres, 'w': w}
        return options


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MainWindowWidget()
    window.show()
    sys.exit(app.exec_())