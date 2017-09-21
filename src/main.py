__author__ = 'Administrator'

import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtCore,QtGui
import numpy as np
import pickle
import os.path

from FileFinder import FileFinderWidget
from REFPlot import REFPlotWidget
from SEGSTable import SEGSTable
from resoluWidget import ResoWidget
from MCRALSMASS import MCRALSMASSPlotDlg
from TICPlot import TICPlot
from MSRTTable import MSRTTableWidget
from MASSDlg import MASSPlotDlg
from SETMARS import SetmarsQDialg
from LibrarySearchDialog import LibrarySearchDialog

from MARS_methods import mars
from NetCDF import netcdf_reader


class MainWindowWidget(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindowWidget, self).__init__()
        QtGui.QMainWindow.__init__(self)

        self.create_menu_toolbar()

        self.fileWidget = FileFinderWidget()
        self.fileWidget.setMaximumWidth(250)
        self.tabWidget = QTabWidget()

        self.refplot = REFPlotWidget()
        self.refplot.setMinimumHeight(200)
        self.refplot.setMaximumHeight(700)
        self.segtable = SEGSTable()
        self.resoluset = ResoWidget()
        self.mcralsmass = MCRALSMASSPlotDlg()
        self.mcralsmass.setMaximumWidth(1180)
        self.updatabutton = QPushButton("Update MSRT")

        self.ticplot = TICPlot()
        self.ticplot.setMinimumHeight(200)
        self.ticplot.setMaximumHeight(700)
        self.msrttable = MSRTTableWidget()
        # self.msrttable.setMaximumWidth(300)
        self.massdlg = MASSPlotDlg()
        self.massdlg.setMaximumWidth(1180)
        self.setmarsbutton = QPushButton('Set MARS')
        self.startmasrbutton = QPushButton('Start MARS')
        self.restartmasrbutton = QPushButton('REStart MARS')
        self.LIBSERCHbutton = QPushButton("LIB search")
        self.exportbutton = QPushButton("EXPORT")

        # self.nextbutton = QPushButton("NEXT")
        # self.prevbutton = QPushButton("PREV")

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
        self.segtable.setFixedWidth(250)
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
        vhbox.addWidget(self.restartmasrbutton)
        vhbox.addWidget(self.LIBSERCHbutton)
        vhbox.addStretch()
        vhbox.addWidget(self.exportbutton)

        # vhbox1 = QVBoxLayout()
        # vhbox1.addStretch()
        # vhbox1.addWidget(self.prevbutton)
        # vhbox1.addWidget(self.nextbutton)
        # vhbox1.addStretch()

        Vlay1 = QHBoxLayout()
        Vlay1.addWidget(self.massdlg)
        Vlay1.setStretchFactor(self.massdlg, 6)
        Vlay1.setStretchFactor(vhbox, 1)
        # Vlay1.addLayout(vhbox1)
        Vlay1.addStretch()
        Vlay1.addLayout(vhbox)
        VComwidget = QWidget()
        VComwidget.setLayout(Vlay1)

        VHQsplitter = QSplitter(Qt.Horizontal, self)
        self.msrttable.setFixedWidth(250)
        VHQsplitter.addWidget(self.msrttable)
        VHQsplitter.addWidget(VComwidget)
        VHQsplitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        VISUALSplitter = QSplitter(Qt.Vertical, self)
        VISUALSplitter.setStretchFactor(1, 3)
        VISUALSplitter.addWidget(self.ticplot)
        VISUALSplitter.addWidget(VHQsplitter)

        self.tabWidget.addTab(ManualSplitter, "RESOLVE")
        self.tabWidget.addTab(VISUALSplitter, "EXTRACT")

        mainSplitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(mainSplitter)
        mainSplitter.addWidget(self.fileWidget)
        mainSplitter.addWidget(self.tabWidget)
        self.setCentralWidget(mainSplitter)

        self.setmarsdlg = SetmarsQDialg()

        self.connect(self.fileWidget, SIGNAL("delete_current"), self.mcralsmass.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.msrttable.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.segtable.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.refplot.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.ticplot.clear_data)
        self.connect(self.fileWidget, SIGNAL("delete_current"), self.massdlg.clear_data)
        self.connect(self.fileWidget, SIGNAL("ref_ncr"), self.resoluset.update_ncr)
        self.connect(self.fileWidget, SIGNAL("sele_file"), self.update_fn)
        self.connect(self.fileWidget, SIGNAL("tic_plot"), self.ticfig)

        self.connect(self.segtable, SIGNAL("delete_segs"), self.refplot.update_segments)
        self.connect(self.segtable, SIGNAL("mass_plot"), self.mcralsmass.add_resolmass)
        self.connect(self.segtable, SIGNAL("select_row"), self.resoluset.updata_index)
        self.connect(self.segtable, SIGNAL("seg_no"), self.resoluset.updata_index)
        self.connect(self.segtable, SIGNAL("updataseg_no"), self.resoluset.updata)
        self.connect(self.segtable, SIGNAL("msrt_list"), self.msrttable.add_msrt)

        self.connect(self.refplot, SIGNAL("MASS_SELECT"), self.mcralsmass.addrawmass)
        self.connect(self.refplot, SIGNAL("delete_SELECT"), self.segtable.undoreply)
        self.connect(self.refplot, SIGNAL("range_SELECT"), self.segtable.add_segs)

        self.connect(self.ticplot, SIGNAL("MASS_SELECT"), self.massdlg.addrawmass)
        self.connect(self.ticplot, SIGNAL("res_plot"), self.getRES)
        self.connect(self.ticplot, SIGNAL("ext_plot"), self.getEXT)

        self.connect(self.msrttable, SIGNAL("updata_com"), self.massdlg.updatamaxc)
        self.connect(self.msrttable, SIGNAL("updata_msrt"), self.update_msrt)
        self.connect(self.msrttable, SIGNAL("masschrom_plot"), self.drug)

        self.connect(self.resoluset, SIGNAL("results"), self.segtable.updata_resulist)
        self.connect(self.setmarsbutton, SIGNAL("clicked()"), self.setparameter)

        # self.connect(self.segtable, SIGNAL("updata_msrt"), self.msrttable.update_msrt)
        self.connect(self.fileWidget, SIGNAL("update_files"), self.update_files)
        self.connect(self.fileWidget, SIGNAL("updata_mars_results"), self.update_mars_results)

        self.updatabutton.clicked.connect(self.addnew_msrt)
        self.massdlg.btprev.on_clicked(self.prevcom)
        self.massdlg.btnext.on_clicked(self.nextcom)
        self.exportbutton.clicked.connect(self.getQUANQUAL_table)
        self.startmasrbutton.clicked.connect(self.getmars_results)
        self.restartmasrbutton.clicked.connect(self.restarmars_results)
        self.LIBSERCHbutton.clicked.connect(self.searchNISTLibrary)

        self.options = self.setmarsdlg.options
        self.files = self.fileWidget.files
        # self.MSRT = self.msrttable.MSRT
        self.MSRT = {}

        self.results = []
        self.idents = []
        self.seleno = 0
        self.refflag = QString()
        self.ticflag = QString()
        self.selefiles = str()
        self.msrtflag = str()
        self.massflag = str()
        self.finish_files = []

        self.resize(1300, 800)
        self.setMinimumWidth(1150)
        self.setMinimumHeight(650)
        self.move(75, 50)
        self.setWindowTitle("MARS -- MS-Assisted Resolution of Signal")

    def searchNISTLibrary(self):
        items = self.msrttable.msrtTable.selectedItems()
        if len(items)/2 == 1:
            row = self.msrttable.msrtTable.row(items[0])
            ms = {'mz':self.MSRT['mz'], 'val':self.MSRT['ms'][row]}
            dialog = LibrarySearchDialog(ms)
            dialog.move(QPoint(50, 50))
            res=dialog.exec_()
            cc = res
        else:
            msgBox = QMessageBox()
            msgBox.setText('Please select one item')
            msgBox.exec_()

    def prevcom(self, event):
        if len(self.massdlg.massplot):
            rtitems = self.msrttable.get_rtitem()
            if self.massdlg.massplot in rtitems:
                pos = rtitems.index(self.massdlg.massplot)
                if pos >= 1:
                    rt = self.MSRT['rt'][pos - 1]
                    self.jude_massplot(pos-1)
            else:
                msgBox = QMessageBox()
                msgBox.setText('not available')
                msgBox.exec_()

    def nextcom(self, event):
        if len(self.massdlg.massplot):
            rtitems = self.msrttable.get_rtitem()
            if self.massdlg.massplot in rtitems:
                pos = rtitems.index(self.massdlg.massplot)
                if len(rtitems) > pos+1:
                    # massplot = rtitems[pos+1]
                    rt = self.MSRT['rt'][pos+1]
                    self.jude_massplot(pos+1)
            else:
                msgBox = QMessageBox()
                msgBox.setText('not available')
                msgBox.exec_()

    def addnew_msrt(self):
        if len(self.segtable.resulist) == 0:
            msgBox = QMessageBox()
            msgBox.setText("NO RESOLVED FILES.")
            msgBox.exec_()
        else:
            self.MSRT = self.segtable.msrtlist()
            # print(len(self.MSRT['rt']))
            # msgBox = QMessageBox()
            # msgBox.setText("MSRT table has updated")
            self.msrttable.add_msrt(self.MSRT)
            # msgBox.exec_()

    def update_msrt(self, msrt, rows):
        self.MSRT = msrt
        if len(rows):
            if len(msrt['rt']) == 0:
                self.results = []
            elif len(rows) == len(self.msrttable.finished):
                self.results = []
            else:
                self.results = self.delete_result(rows)
            self.msrttable.update_resulist(self.results)

    def delete_result(self, rows):
        results = []
        for i in range(0, len(self.files['fn'])-1):
            old_res = self.results[i]
            chrs = old_res['chrs']
            orcs = old_res['orc']
            segs = old_res['segs']
            tzs = old_res['tzs']
            areas = old_res['areas']
            highs = old_res['highs']
            rts = old_res['rts']
            for i, v in enumerate(rows):
                chrs.pop(v)
                segs = np.delete(segs, v, axis=0)
                tzs = np.delete(tzs, v, axis=0)
                areas = np.delete(areas, v)
                highs = np.delete(highs, v)
                orcs.pop(v)
            results.append({'chrs': chrs, 'segs': segs, 'tzs': tzs, 'areas': areas,
                             'highs': highs, 'rts': rts, 'orc': orcs})
        return results

    def update_files(self, files):
        self.files = files

    def update_mars_results(self, file):
        for fn in file:
            if fn in self.finish_files:
                ind = self.finish_files.index(fn)-1
                self.finish_files.pop(ind)
                self.results.pop(ind)

    def ticfig(self, fn, row):
        file = self.files['files'][row]
        ncr = netcdf_reader(file, bmmap=False)
        # self.refflag = str(fn)
        if '&' in fn:
            self.refplot.add_tic(ncr, fn)
            self.refflag = fn
            self.resoluset.update_ncr(ncr)
        else:
            self.ticplot.add_tic(ncr, fn)
            self.ticflag = fn
            massplot = self.massdlg.massplot
            files = self.files['fn']
            row = files.index(self.ticflag)
            rtitems = self.msrttable.get_rtitem()
            if len(massplot):
                num = rtitems.index(massplot)
                rt = self.MSRT['rt'][num]
                self.mainaddresol(num, row)

    def jude_massplot(self, num):
        if len(self.ticflag):
            files = self.files['fn']
            row = files.index(self.ticflag)
            self.mainaddresol(num, row)
        else:
            row = -1
            self.mainaddresol(num, row)

    def mainaddresol(self, num, row):
        rt = self.MSRT['rt'][num]
        ms = self.MSRT['ms'][num]
        mz = self.MSRT['mz']
        chrom = {}
        if num in self.msrttable.finished:
            nnum = np.searchsorted(self.msrttable.finished, num)
            if row >= 0:
                res = self.results[row-1]
                file = self.files['files'][row]
                rts = netcdf_reader(file, bmmap=False).tic()
                tz = res['tzs'][nnum,:]
                seg = res['segs'][nnum,:]
                tzs = rts['rt'][tz[0]:tz[1]]
                segs = rts['rt'][seg[0]:seg[1]]
                orc = res['orc'][nnum]
                chr = res['chrs'][nnum]
                chrom = {'rts': tzs, 'chrom': chr, 'segs': segs, 'orc': orc}
        self.massdlg.addresolmass(rt, ms, mz, num, chrom, self.ticflag)

    def update_fn(self, fn, no):
        self.selefiles = fn
        self.seleno = no

    def drug(self, num):
        if len(self.results):
            self.jude_massplot(num)
        else:
            row = -1
            self.mainaddresol(num, row)

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
        self.setmarsdlg.setModal(True)
        self.setmarsdlg.move(QPoint(50, 50))
        self.setmarsdlg.show()
        self.setmarsdlg.exec_()
        self.options = self.setmarsdlg.options

    def restarmars_results(self):
        self.results = []
        self.msrttable.update_resulist(self.results)
        self.getmars_results()

    def getmars_results(self):
        if len(self.MSRT) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please input MSRT.")
            msgBox.exec_()
        elif len(self.files) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please loading files.")
            msgBox.exec_()
        else:
            if len(self.results) == 0:
                MSRT = self.MSRT
                files = self.files #['files']
                fns = self.files['fn']
                flag = 0
                fflag = 0
            else:
                if len(self.MSRT['rt']) == len(self.results[0]['tzs']):
                    MSRT = self.MSRT
                    flag = 0
                else:
                    flag = 1
                    new_rt = []
                    new_ms = []
                    vsel = []
                    for i in range(0, len(self.MSRT['rt'])):
                        if i not in self.msrttable.finished:
                            vsel.append(i)
                            new_rt.append(self.MSRT['rt'][i])
                            new_ms.append(self.MSRT['ms'][i])
                    MSRT = {'ms': new_ms, 'rt': new_rt}

                if self.files['fn'] == self.finish_files:
                    fils = self.files['files']
                    fns = self.files['fn']
                    fflag = 0
                else:
                    fflag = 1
                    fns = []
                    fils = []
                    for i, fn in enumerate(self.files['fn']):
                        if str(fn) not in self.finish_files:
                            fns.append(fn)
                            fils.append(self.files['files'][i])
                if flag == 0 and fflag == 0:
                    return
                dir = self.files['dir']
                files = {'fn': fns, 'files': fils, 'dir': dir}

            progressDialog = QProgressDialog(self)
            progressDialog.setWindowModality(Qt.WindowModal)
            progressDialog.setMinimumDuration(1)
            progressDialog.setWindowTitle('waiting...')
            progressDialog.setCancelButtonText('cancle')
            progressDialog.setRange(1, len(self.files['files']))

            if flag == 1 and fflag == 1:
                results1 = []
                results2 = []
                for ind, fn in enumerate(self.files['files']):
                    QApplication.processEvents()
                    # progressDialog.setLabelText(str(ind+1)+'/' + str(len(self.files['files']))+'th files processing')
                    # progressDialog.setValue(ind + 1)
                    if '&' not in self.files['fn'][ind]:
                        ncr = netcdf_reader(fn, bmmap=False)
                        if str(self.files['fn'][ind]) in self.finish_files:
                            result = mars(ncr, MSRT, self.options)
                            results1.append(result)
                        else:
                            result = mars(ncr, self.MSRT, self.options)
                            results2.append(result)
                    print(ind)
                    QApplication.processEvents()
                    progressDialog.setLabelText(str(ind+1)+'/' + str(len(self.files['files']))+'th files processing')
                    progressDialog.setValue(ind + 1)
                    if progressDialog.wasCanceled():
                        return
                self.results = self.combine_result(results1, vsel)
                self.results.extend(results2)
            # elif flag == 0 and fflag == 0:
            #     msgBox = QMessageBox()
            #     msgBox.setText("Files have finished")
            #     msgBox.exec_()
            else:
                results = []
                for ind, fn in enumerate(files['files']):
                    QApplication.processEvents()
                    progressDialog.setLabelText(str(ind+1)+'/' + str(len(fns))+'th files processing')
                    progressDialog.setValue(ind + 1)
                    if '&' not in files['fn'][ind]:
                        ncr = netcdf_reader(fn, bmmap=False)
                        result = mars(ncr, MSRT, self.options)
                        results.append(result)
                    print(ind)
                    QApplication.processEvents()
                    progressDialog.setLabelText(str(ind+1)+'/' + str(len(fns))+'th files processing')
                    progressDialog.setValue(ind + 1)
                    if progressDialog.wasCanceled():
                        return
                if flag == 1 and fflag == 0:
                    self.results = self.combine_result(results, vsel)
                elif flag == 0:
                    self.results.extend(results)
            self.finish_files = self.files['fn']
            self.msrttable.update_resulist(self.results)

    def combine_result(self, results, vsel):
        fin_results = []
        for i in range(0, len(results)):
            old_res = self.results[i]
            new_res = results[i]
            chrs = old_res['chrs']
            orcs = old_res['orc']
            segs = old_res['segs']
            tzs = old_res['tzs']
            areas = old_res['areas']
            highs = old_res['highs']
            rts = old_res['rts']

            for j, v in enumerate(vsel):
                chrs.insert(v, new_res['chrs'][j])
                segs = np.insert(segs, v, new_res['segs'][j,:], axis=0)
                tzs = np.insert(tzs, v, new_res['tzs'][j,:], axis=0)
                areas = np.insert(areas, v, new_res['areas'][j])
                highs = np.insert(highs, v, new_res['highs'][j])
                # rts = np.insert(rts, v, new_res['rts'])
                # rts.insert(v, new_res['rts'][j])
                orcs.insert(v, new_res['orc'][j])
            new_old = {'chrs': chrs, 'segs': segs, 'tzs': tzs, 'areas': areas, 'highs': highs, 'rts':rts, 'orc': orcs}
            fin_results.append(new_old)
        return fin_results

    def getQUANQUAL_table(self):
        if len(self.results) >= 1:
            if len(self.idents) == 0:
                # name = QtGui.QFileDialog.getSaveFileName(self, "Find Files", QDir.currentPath())
                # name = QtGui.QFileDialog.getSaveFileNameAndFilter()
                file_formats = "txt file (*.txt);;"
                #JPG File (*.jpeg *.jpg);;

                path, selected_filter = QtGui.QFileDialog.getSaveFileNameAndFilter(self, "Find Files", ".",
                                                                                   file_formats)
                if not path:
                    return

                pre_vec = ['comID', 'RT.ref(min)']
                com = len(self.results[0]['chrs'])

                file_str = []
                ref_no = 0
                for i, fn in enumerate(self.files['fn']):
                    if '&' in fn:
                        ref_na = fn.replace("&", "")
                        if ref_na in self.files['fn']:
                            ref_no = 1
                            ix = self.files['fn'][::-1].index(ref_na)
                            fn_str = str(fn)
                            file_str.append(fn_str.replace(".cdf", ""))
                    else:
                        fn_str = str(fn)
                        file_str.append(fn_str.replace(".cdf", ""))
                pre_vec.extend(file_str)
                fn_vec = np.array(pre_vec, ndmin=2)
                # Dir = str(self.files['dir'])

                area = np.zeros((len(self.files['fn'])+1+ref_no, com))
                area[0, :] = range(1, com+1)
                area[1, :] = np.array(self.MSRT['rt'])[self.msrttable.finished]
                if ref_no == 1:
                    area[2, :] = self.results[::-1][ix]['areas']
                for ind, result in enumerate(self.results):
                    area[ind+2+ref_no, :] = result['areas']
                DAT = np.vstack((fn_vec, area.T))
                path1 = str(path)
                np.savetxt(path1, DAT, delimiter=",", fmt="%s")
                self.get_msp(self.MSRT)

    def get_msp(self, MSRT):
        mz = MSRT['mz']
        path = self.files['dir']+'\qual.msp'
        msp = open(path, 'w')
        for i in range(0, len(MSRT['rt'])):
            rt = MSRT['rt'][i]
            ms = MSRT['ms'][i]
            msp.write("%s %f \n"%("Name: rt", rt))
            msp.write("%s %f \n"%("Num peaks:", len(ms)))
            for j in range(0, len(ms)):
                msp.write("%2.5f \t %f \n" % (mz[j], ms[j]))
        msp.close()

    def Qstr2Str(self, fns):
        dir = str(fns['dir'])
        files = []
        fn = []
        for i in range(0, len(fns['files'])):
            # fn.append(unicode(fns['fn'][i]))
            fn.append(str(fns['fn'][i]))
            files.append(str(fns['files'][i]))
        return {'fn': fn, 'files': files, 'dir': dir}

    def get_project(self):
        fns = self.fileWidget.files
        files = self.Qstr2Str(fns)
        MSRT = self.msrttable.MSRT
        results = self.results
        options = self.setmarsdlg.options
        segments = self.segtable.segments
        ref_seg = self.refplot.segments
        resu_results = self.segtable.resulist
        finished = self.msrttable.finished
        finish_files = self.finish_files
        return files, results, MSRT, options, segments, ref_seg, resu_results, finished, finish_files

    def saveprojects(self):
        files, results, MSRT, options, segments, ref_seg, resu_results, finished, finish_files = self.get_project()
        projection_data = {'files': files, 'MSRT': MSRT, 'options': options, 'segments': segments, 'ref_seg': ref_seg,
                           'results': results, 'resu_results': resu_results, 'finished': finished, 'finish_files': finish_files}
        # Dir = QDir.currentPath()
        Dir = files['dir']
        DATAF = open(str(Dir) + '\MARS_Project.pkl', 'w')
        pickle.dump(projection_data, DATAF)
        DATAF.close()

    def saveprojectsas(self):
        name = QtGui.QFileDialog.getSaveFileName(self, "Find Files", QDir.currentPath())
        if len(name):
            files, results, MSRT, options, segments, ref_seg, resu_results, finished, finish_files = self.get_project()
            projection_data = {'files': files, 'MSRT': MSRT, 'options': options, 'segments': segments, 'ref_seg': ref_seg,
                               'results': results, 'resu_results': resu_results, 'finished': finished, 'finish_files': finish_files}
            # DATAF = open(str(Dir) + '\MARS_Project.pkl', 'w')
            DATAF = open(name, 'w')
            pickle.dump(projection_data, DATAF)
            DATAF.close()

    def OpenProjection(self):
        name = QtGui.QFileDialog.getOpenFileName(self, "Loading Project", QDir.currentPath())
        # path = QtGui.QFileDialog.get
        if len(name):
            if len(self.files['fn']) >= 2:
                reply = QMessageBox.warning(self, "Warning...",
                                        "Replaced Current Project ?",
                                        QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    pkl_file = open(str(name))
                    Pdata = pickle.load(pkl_file)
                    self.clear_curr()
                    self.updata_inputs(Pdata, name)
            else:
                pkl_file = open(str(name))
                Pdata = pickle.load(pkl_file)
                self.updata_inputs(Pdata, name)
        else:
            msgBox = QMessageBox()
            msgBox.setText("NO selected Project.")
            msgBox.exec_()

    def updata_inputs(self, pdata, path):
        self.results = []
        self.idents = []
        self.seleno = 0
        self.refflag = QString()
        self.ticflag = QString()
        self.selefiles = str()
        self.msrtflag = str()
        self.massflag = str()
        self.MSRT = pdata['MSRT']

        self.files = pdata['files']
        files = []
        tt = os.path.isdir(self.files['dir'])
        if not os.path.isdir(self.files['dir']):
            dir = str(self.fileWidget.directoryComboBox.currentText())
        else:
            dir = self.files['dir']

        for fn in self.files['fn']:
            if '&' in fn:
                fnl = fn[1:]
            else:
                fnl = fn
            if os.path.isfile(dir+'/'+fnl):
                files.append(dir+'/'+fnl)
        if len(files) == len(self.files['files']):
            self.files['dir'] = dir
            self.files['files'] = files
        elif len(files):
            msgBox = QMessageBox()
            msgBox.setText("CDF Files not complete")
            msgBox.exec_()
            return
        else:
            msgBox = QMessageBox()
            msgBox.setText("CDF Files not exist in Dir")
            msgBox.exec_()
            return

        self.options = pdata['options']
        self.results = pdata['results']
        self.finish_files = pdata['finish_files']
        ref_seg = pdata['ref_seg']
        segments = pdata['segments']
        resu_results = pdata['resu_results']
        ncr = netcdf_reader(self.files['files'][0], bmmap=False)
        finished = pdata['finished']

        self.fileWidget.loading(self.files)
        self.segtable.loading(segments, resu_results)
        self.refplot.loading(ref_seg, ncr, self.files['fn'][0])
        self.msrttable.loading(self.MSRT, finished)
        self.ticplot.loading()
        self.resoluset.loading(segments, ncr)
        self.massdlg.loading()
        self.mcralsmass.loading()
        self.setmarsdlg.loading(self.options)

    def clear_curr(self):
        return

    def create_menu_toolbar(self):
        # saveOpenAction = QtGui.QAction(QtGui.QIcon('images/save.png'), '&Save', self)
        # saveOpenAction.setShortcut('Ctrl+S')
        # saveOpenAction.setStatusTip('Save results')
        # saveOpenAction.triggered.connect(self.Save)

        # saveasOpenAction = QtGui.QAction(QtGui.QIcon('images/saveas.png'), '&Save as', self)
        # saveasOpenAction.setShortcut('Ctrl+A')
        # saveasOpenAction.setStatusTip('Save results as')
        # saveasOpenAction.triggered.connect(self.Saveas)

        saveProjectionAction = QtGui.QAction(QtGui.QIcon('images/saveprojection.png'), '&Save project', self)
        saveProjectionAction.setShortcut('Ctrl+S')
        saveProjectionAction.setStatusTip('Save projection')
        saveProjectionAction.triggered.connect(self.saveprojects)

        saveProjectionasAction = QtGui.QAction(QtGui.QIcon('images/saveprojectionas.png'), '&Save project as', self)
        saveProjectionasAction.setShortcut('Ctrl+N')
        saveProjectionasAction.setStatusTip('Save projection as')
        saveProjectionasAction.triggered.connect(self.saveprojectsas)

        OpenProjectionAction = QtGui.QAction(QtGui.QIcon('images/openprojection.png'), '&Loading project', self)
        OpenProjectionAction.setShortcut('Ctrl+L')
        OpenProjectionAction.setStatusTip('Open saved projection')
        OpenProjectionAction.triggered.connect(self.OpenProjection)

        ExitAction = QtGui.QAction(QtGui.QIcon('images/save.png'), '&Exit', self)
        ExitAction.setShortcut('Ctrl+E')
        ExitAction.setStatusTip('Exit')
        ExitAction.connect(ExitAction, SIGNAL('triggered()'), QtGui.qApp, SLOT('quit()'))

        aboutme = QtGui.QAction('About me', self)
        updata = QtGui.QAction('Check for updata', self)

        menubar = self.menuBar()
        file = menubar.addMenu('&File')
        # file.addAction(saveOpenAction)
        # file.addAction(saveasOpenAction)
        file.addAction(OpenProjectionAction)
        file.addAction(saveProjectionAction)
        file.addAction(saveProjectionasAction)
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