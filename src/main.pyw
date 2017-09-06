__author__ = 'Administrator'

import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
import numpy as np
import pickle

from FileFinder import FileFinderWidget
from REFPlot import REFPlotWidget
from SEGSTable import SEGSTable
from resoluWidget import ResoWidget
from MCRALSMASS import MCRALSMASSPlotDlg
from TICPlot import TICPlot
from MSRTTable import MSRTTableWidget
from MASSDlg import MASSPlotDlg
from SETMARS import SetmarsQDialg

from MARS_methods import mars
from NetCDF import netcdf_reader


class MainWindowWidget(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindowWidget, self).__init__()
        QtGui.QMainWindow.__init__(self)

        self.create_menu_toolbar()

        self.fileWidget = FileFinderWidget()
        # self.fileWidget.setMinimumWidth(200)
        self.fileWidget.setMaximumWidth(300)
        self.tabWidget = QTabWidget()
        ManualWidget = QWidget()
        VisualWidget = QWidget()

        self.refplot = REFPlotWidget()
        self.refplot.setMinimumHeight(200)
        self.refplot.setMaximumHeight(700)
        self.segtable = SEGSTable()
        self.resoluset = ResoWidget()
        self.mcralsmass = MCRALSMASSPlotDlg()
        self.mcralsmass.setMaximumWidth(1050)
        self.updatabutton = QPushButton("Update MSRT")

        self.ticplot = TICPlot()
        self.ticplot.setMinimumHeight(200)
        self.ticplot.setMaximumHeight(700)
        self.msrttable = MSRTTableWidget()
        # self.msrttable.setMaximumWidth(300)
        self.massdlg = MASSPlotDlg()
        self.massdlg.setMaximumWidth(1050)
        self.setmarsbutton = QPushButton('Set MARS')
        self.startmasrbutton = QPushButton('Start MARS')
        self.LIBSERCHbutton = QPushButton("LIB search")
        self.exportbutton = QPushButton("EXPORT")

        self.nextbutton = QPushButton("NEXT")
        self.prevbutton = QPushButton("PREV")

        # hhbox = QHBoxLayout()
        # hhbox.addStretch()
        # hhbox.addWidget(self.updatabutton)
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.resoluset)
        vbox.addStretch()
        vbox.addWidget(self.updatabutton)
        hbox.addStretch()
        hbox.addLayout(vbox)

        Vlay = QHBoxLayout()
        Vlay.addWidget(self.mcralsmass)
        Vlay.addLayout(hbox)
        Vlay.setStretchFactor(self.mcralsmass, 6)
        Vlay.setStretchFactor(hbox, 1)
        Comwidget = QWidget()
        Comwidget.setLayout(Vlay)

        HQsplitter = QSplitter(Qt.Horizontal, self)
        self.segtable.setFixedWidth(300)
        # self.segtable.setMaximumWidth(300)
        # HQsplitter.setStretchFactor(1, 6)
        HQsplitter.addWidget(self.segtable)
        HQsplitter.addWidget(Comwidget)
        HQsplitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ManualSplitter = QSplitter(Qt.Vertical, self)
        ManualSplitter.setStretchFactor(2, 3)
        ManualSplitter.setOpaqueResize(True)
        ManualSplitter.addWidget(self.refplot)
        ManualSplitter.addWidget(HQsplitter)

        vhbox = QVBoxLayout()
        # vhbox.addStretch()
        vhbox.addWidget(self.setmarsbutton)
        vhbox.addWidget(self.startmasrbutton)
        vhbox.addWidget(self.LIBSERCHbutton)
        vhbox.addStretch()
        vhbox.addWidget(self.exportbutton)


        vhbox1 = QVBoxLayout()
        vhbox1.addStretch()
        vhbox1.addWidget(self.prevbutton)
        vhbox1.addWidget(self.nextbutton)
        vhbox1.addStretch()


        Vlay1 = QHBoxLayout()
        Vlay1.addWidget(self.massdlg)
        Vlay1.setStretchFactor(self.massdlg, 6)
        Vlay1.setStretchFactor(vhbox, 1)
        Vlay1.addLayout(vhbox1)
        Vlay1.addStretch()
        Vlay1.addLayout(vhbox)
        VComwidget = QWidget()
        VComwidget.setLayout(Vlay1)

        VHQsplitter = QSplitter(Qt.Horizontal, self)
        self.msrttable.setFixedWidth(300)
        VHQsplitter.addWidget(self.msrttable)
        VHQsplitter.addWidget(VComwidget)
        VHQsplitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        VISUALSplitter = QSplitter(Qt.Vertical, self)
        VISUALSplitter.setStretchFactor(1, 3)
        VISUALSplitter.addWidget(self.ticplot)
        VISUALSplitter.addWidget(VHQsplitter)

        self.tabWidget.addTab(ManualSplitter, "Manual")
        self.tabWidget.addTab(VISUALSplitter, "Visual")

        mainSplitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(mainSplitter)
        mainSplitter.addWidget(self.fileWidget)
        mainSplitter.addWidget(self.tabWidget)
        self.setCentralWidget(mainSplitter)

        # self.resize(1200, 900)
        # self.setMinimumWidth(1300)
        # self.setMinimumHeight(900)
        self.setWindowTitle("MARS -- MS-Assisted Resolution of Signal")

        self.connect(self.fileWidget, SIGNAL("delete_current"), self.segtable.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.refplot.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.mcralsmass.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.msrttable.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.ticplot.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.massdlg.clear_data)
        self.connect(self.fileWidget, SIGNAL("tic_plot"), self.ticfig)
        self.connect(self.fileWidget, SIGNAL("sele_file"), self.update_fn)

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
        self.connect(self.ticplot, SIGNAL("res_plot"), self.getRES)
        self.connect(self.ticplot, SIGNAL("ext_plot"), self.getEXT)

        self.connect(self.msrttable, SIGNAL("masschrom_plot"), self.drug)
        self.connect(self.msrttable, SIGNAL("updata_com"), self.massdlg.updatamaxc)
        self.connect(self.msrttable, SIGNAL("updata_msrt"), self.update_msrt)

        self.connect(self.resoluset, SIGNAL("results"), self.segtable.updata_resulist)
        self.updatabutton.clicked.connect(self.obmsrt)
        self.connect(self.setmarsbutton, SIGNAL("clicked()"), self.setparameter)

        self.connect(self.fileWidget, SIGNAL("update_files"), self.update_files)
        self.connect(self.segtable, SIGNAL("updata_msrt"), self.msrttable.update_msrt)

        self.prevbutton.clicked.connect(self.prevcom)
        self.nextbutton.clicked.connect(self.nextcom)
        self.startmasrbutton.clicked.connect(self.getmars_results)
        self.exportbutton.clicked.connect(self.getQUANQUAL_table)

        self.idents = []
        self.seleno = 0
        self.selefiles = str()
        self.results = []
        self.options = {'pw': 50, 'thres': 0.8, 'w': 3}

        self.files = self.fileWidget.updata_files()
        self.refflag = str()
        self.ticflag = str()

        self.MSRT = {}
        self.msrtflag = str()
        self.massflag = str()

    # def nextcom(self):
    #     if self.ticflag:
    #         files = self.files['fn']
    #         row = files.index(self.ticflag)
    #         rtitems = self.msrttable.get_rtitem()
    #         num = rtitems.index(self.massdlg.massplot)
    #         if len(rtitems)>=num+2:
    #             self.mainaddresol(num+1, row)

    def prevcom(self):
        if len(self.massdlg.massplot):
            rtitems = self.msrttable.get_rtitem()
            if self.massdlg.massplot in rtitems:
                pos = rtitems.index(self.massdlg.massplot)
                if pos >= 1:
                    massplot = rtitems[pos-1]
                    self.jude_massplot(massplot)
            else:
                msgBox = QMessageBox()
                msgBox.setText('not available')
                msgBox.exec_()

    def nextcom(self):
        if len(self.massdlg.massplot):
            rtitems = self.msrttable.get_rtitem()
            if self.massdlg.massplot in rtitems:
                pos = rtitems.index(self.massdlg.massplot)
                if len(rtitems) > pos+1:
                    massplot = rtitems[pos+1]
                    self.jude_massplot(massplot)
            else:
                msgBox = QMessageBox()
                msgBox.setText('not available')
                msgBox.exec_()

        # if self.ticflag:
        #     files = self.files['fn']
        #     row = files.index(self.ticflag)
        #     rtitems = self.msrttable.get_rtitem()
        #     num = rtitems.index(self.massdlg.massplot)
        #     if num >= 1:
        #         self.mainaddresol(num-1, row)
        # else:
        #     if len(self.massdlg.massplot):
        #         rtitems = self.msrttable.get_rtitem()
        #         num = rtitems.index(self.massdlg.massplot)
        #         if self.massdlg.massplot not in rtitems:
        #             msgBox = QMessageBox()
        #             msgBox.setText("The MS was deleted.")
        #             msgBox.exec_()
        #         else:
        #             num = rtitems.index(self.massdlg.massplot)
        #             self.mainaddresol(num, row)

    def obmsrt(self):
        self.MSRT = self.segtable.msrtlist()
        self.msrttable.add_msrt(self.MSRT)

    def update_msrt(self, msrt, results):
        self.MSRT = msrt
        self.results = results

    def update_files(self, files):
        self.files = files

    def ticfig(self, fn, row):
        file = self.files['files'][row]
        ncr = netcdf_reader(file)
        # self.refflag = str(fn)
        if '&' in fn:
            self.refplot.add_tic(ncr, fn)
            self.refflag = str(fn)
        else:
            self.ticplot.add_tic(ncr, fn)
            self.ticflag = str(fn)
            self.jude_massplot(self.massdlg.massplot)

    def jude_massplot(self, massplot):
        if len(massplot):
            rtitems = self.msrttable.get_rtitem()
            if len(self.ticflag):
                if self.massdlg.massplot not in rtitems:
                    msgBox = QMessageBox()
                    msgBox.setText("The MS was deleted.")
                    msgBox.exec_()
                else:
                    files = str(self.files['fn'])
                    row = files.index(self.ticflag)
                    num = rtitems.index(massplot)
                    self.mainaddresol(num, row)
            else:
                num = rtitems.index(massplot)
                row = -1
                self.mainaddresol(num, row)

    def mainaddresol(self, num, row):
        ms = self.MSRT['ms'][num]
        rt = self.MSRT['rt'][num]
        mz = self.MSRT['mz']
        chrom = {}
        if len(self.results) >= 1:
            if row >= 0:
                res = self.results[row]
                tzs = res['rts'][res['tzs']]
                segs = res['rts'][res['tzs']]
                orc = res['orc']
                chr = res['chrs']
                chrom = {'rts': tzs, 'chrom': chr, 'segs': segs, 'orc': orc}
        self.massdlg.addresolmass(rt, ms, mz, num, chrom, self.ticflag)

    def update_fn(self, fn, no):
        self.selefiles = fn
        self.seleno = no

    def drug(self, rt, ms, mz, row):
        if len(self.results) == 0:
            chrom = dict()
        else:
            res = self.results[row]
            tzs = res['rts'][res['tzs']]
            segs = res['rts'][res['tzs']]
            orc = res['orc']
            chr = res['chrs']
            chrom = {'rts': tzs, 'chrom': chr, 'orc': orc, 'segs':segs}
        self.massdlg.addresolmass(rt, ms, mz, row, chrom, self.ticflag)

    # def maschr(self, ):

    def getEXT(self, no):
        if no == 0:
            msgBox = QMessageBox()
            msgBox.setText("NO selected files.")
            msgBox.exec_()
            if len(self.results) == 0:
                msgBox = QMessageBox()
                msgBox.setText("NO RESOLVED RESULTS")
                msgBox.exec_()
            else:
                extchrom = np.zeros(self.rt.shape)
                for ind, V in enumerate(self.results):
                    tz = V['tz']
                    extchrom[tz[1]:tz[2]] = V['chrom']
                self.ticplot.add_ext(extchrom)

    def getRES(self, no):
        if no == 0:
            msgBox = QMessageBox()
            msgBox.setText("NO selected files.")
            msgBox.exec_()
            if len(self.results) == 0:
                msgBox = QMessageBox()
                msgBox.setText("NO RESOLVED RESULTS")
                msgBox.exec_()
            else:
                reschrom = np.zeros(self.rt.shape)
                for ind, V in enumerate(self.results):
                    tz = V['tz']
                    reschrom[tz[1]:tz[2]] = V['chrom']
                self.ticplot.add_res(reschrom)


    def setparameter(self):
        self.setmarsdlg = SetmarsQDialg(self.options)
        self.setmarsdlg.setModal(True)
        self.setmarsdlg.move(QPoint(50, 50))
        self.setmarsdlg.show()
        self.setmarsdlg.exec_()
        self.options = self.setmarsdlg.options

    def updata_inputs(self):
        self.results = []
        self.files = self.fileWidget.get_files()
        self.MSRT = self.msrttable.get_MSRT()

    def getmars_results(self):
        self.updata_inputs()
        if len(self.MSRT) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please input MSRT.")
            msgBox.exec_()
        elif len(self.files) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please loading files.")
            msgBox.exec_()
        else:
            files = self.files['files']
            fns = self.files['fn']

            progressDialog = QProgressDialog(self)
            progressDialog.setWindowModality(Qt.WindowModal)
            progressDialog.setMinimumDuration(1)
            progressDialog.setWindowTitle('waiting...')

            progressDialog.setCancelButtonText('cancle')
            progressDialog.setRange(0, len(fns))

            for ind, fn in enumerate(files):
                QApplication.processEvents()
                progressDialog.setLabelText(str(ind+1)+'/' + str(len(fns))+'th files processing')
                progressDialog.setValue(ind)
                QThread.msleep(100)
                if '&' not in fn[ind]:
                    ncr = netcdf_reader(fn)
                    result = mars(ncr, self.MSRT, self.options)
                    self.results.append(result)
                if progressDialog.wasCanceled():
                    return
        self.msrttable.resulist(self.results)

    def getQUANQUAL_table(self):
        if len(self.results) >= 1:
            if len(self.idents) == 0:
                pre_vec = ['comID', 'RT.ref(min)']
                file_str = []
                ref_no=0
                for i, fn in enumerate(self.files['filenames']):
                    if '&' not in fn:
                        fn_str = str(fn)
                        file_str.append(fn_str.replace(".cdf", ""))
                    else:
                        ref_no = 1
                pre_vec.extend(file_str)
                fn_vec = np.array(pre_vec, ndmin=2)
                Dir = str(self.files['Dir'])

                area = np.zeros((len(self.files['filenames'])+2-ref_no, len(self.MSRT['rt'])))
                area[0, :] = range(1, len(self.MSRT['rt'])+1)
                area[1, :] = self.MSRT['rt']
                for ind, result in enumerate(self.results):
                    area[ind+2, :] = result['areas']
                DAT = np.vstack((fn_vec, area.T))
                np.savetxt('test11.txt', DAT, delimiter=",", fmt="%s")

    def saveproject(self):
        self.updata_inputs()
        projection_data = {'files': self.files, 'MSRT': self.MSRT, 'options': self.options,
                           'results': self.results}
        DATAF = open(self.files['directory'] + 'marsProjection.pkl', 'w')
        pickle.dump(projection_data, DATAF)
        DATAF.close()

    def saveProjectionas(self):
        directoryQDialg = QFileDialog.getSaveFileName()
        savedirectory = []
        self.updata_inputs()
        projection_data = {'files': self.files, 'MSRT': self.MSRT, 'options': self.options,
                           'results': self.results}
        DATAF = open(savedirectory + 'marsProjection.pkl', 'w')
        pickle.dump(projection_data, DATAF)
        DATAF.close()

    # def saveprojectionas(self):
    def create_menu_toolbar(self):
        saveOpenAction = QtGui.QAction(QtGui.QIcon('images/save.png'), '&Save', self)
        saveOpenAction.setShortcut('Ctrl+S')
        saveOpenAction.setStatusTip('Save results')
        # saveOpenAction.triggered.connect(self.Save)

        saveasOpenAction = QtGui.QAction(QtGui.QIcon('images/saveas.png'), '&Save as', self)
        saveasOpenAction.setShortcut('Ctrl+A')
        saveasOpenAction.setStatusTip('Save results as')
        # saveasOpenAction.triggered.connect(self.Saveas)

        saveProjectionAction = QtGui.QAction(QtGui.QIcon('images/saveprojection.png'), '&Save projection', self)
        saveProjectionAction.setShortcut('Ctrl+P')
        saveProjectionAction.setStatusTip('Save projection')
        # saveProjectionAction.triggered.connect(self.Saveprojection)

        saveProjectionasAction = QtGui.QAction(QtGui.QIcon('images/saveprojectionas.png'), '&Save projection as', self)
        saveProjectionasAction.setShortcut('Ctrl+M')
        saveProjectionasAction.setStatusTip('Save projection as')
        # saveProjectionasAction.triggered.connect(self.saveProjectionas)

        OpenProjectionAction = QtGui.QAction(QtGui.QIcon('images/openprojection.png'), '&Open projection', self)
        OpenProjectionAction.setShortcut('Ctrl+O')
        OpenProjectionAction.setStatusTip('Open saved projection')
        # OpenProjectionAction.triggered.connect(self.OpenProjection)

        ExitAction = QtGui.QAction(QtGui.QIcon('images/save.png'), '&Exit', self)
        ExitAction.setShortcut('Ctrl+E')
        ExitAction.setStatusTip('Exit')
        ExitAction.connect(ExitAction, SIGNAL('triggered()'), QtGui.qApp, SLOT('quit()'))

        aboutme = QtGui.QAction('About me', self)
        updata = QtGui.QAction('Check for updata', self)

        menubar = self.menuBar()
        file = menubar.addMenu('&File')
        file.addAction(saveOpenAction)
        file.addAction(saveasOpenAction)
        file.addAction(saveProjectionAction)
        file.addAction(saveProjectionasAction)
        file.addAction(OpenProjectionAction)
        file.addAction(ExitAction)
        # resolu = menubar.addMenu('&Resolution')
        # resolu.addAction(selectreference)
        # resolu.addAction(resolution)
        # marsmain = menubar.addMenu('&MARS main')
        # marsmain.addAction(setpara)
        # marsmain.addAction(start)
        # marsmain.addAction(stop)
        helpme = menubar.addMenu('&Help')
        helpme.addAction(aboutme)
        helpme.addAction(updata)

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MainWindowWidget()
    window.show()
    sys.exit(app.exec_())