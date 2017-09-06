__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from NetCDF import netcdf_reader
import numpy as np
import sys

class SEGSTable(QtGui.QWidget):
    def __init__(self, parent=None):
        super(SEGSTable, self).__init__(parent)

        self.segments = []
        self.MS = np.array([])
        self.RT = np.array([])
        self.rtlist = []
        self.rtsor = []
        self.createsegsTable()
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.segsTable)
        self.setLayout(mainLayout)
        self.connect(self.segsTable, SIGNAL("itemClicked (QTableWidgetItem*)"), self.outSelect)

        self.setWindowTitle("MSRT")
        # self.resize(300, 600

    def createsegsTable(self):
        self.segsTable = QtGui.QTableWidget(0, 3)
        self.segsTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.segsTable.setHorizontalHeaderLabels(("Seg", "PC", "R2"))
        self.segsTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
        self.segsTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.segsTable.horizontalHeader().setResizeMode(2, QtGui.QHeaderView.Stretch)
        self.segsTable.verticalHeader().hide()
        self.segsTable.setShowGrid(True)
        self.segsTable.customContextMenuRequested.connect(self.showContextMenu)

        self.contextMenu = QtGui.QMenu(self)
        self.actionshow = self.contextMenu.addAction('Showmcr')
        self.actiondelete = self.contextMenu.addAction('Delete')
        self.actionshow.triggered.connect(self.showmcr)
        self.actiondelete.triggered.connect(self.delete)

    def add_segs(self, ncr, (indmin, indmax)):
        self.ncr = ncr
        seg = (indmin, indmax)
        self.segments.append(seg)
        rt = ncr.scan_acquisition_time
        segmin = [rt[indmin]/60, rt[indmax]/60]

        nd_segments = np.array(self.segments)
        starts = nd_segments[:, 0]
        segsorts = np.sort(starts)
        row = np.nonzero(segsorts == starts[-1])
        insertrow = np.array(row, dtype='int')
        self.sortseg(self.segments)

        roundseg = [round(segmin[0], 3), round(segmin[1], 3)]
        segItem = QtGui.QTableWidgetItem(str(roundseg))
        segItem.setFlags(segItem.flags() ^ QtCore.Qt.ItemIsEditable)
        self.segsTable.insertRow(insertrow)
        self.segsTable.setItem(insertrow, 0, segItem)
        self.emit(SIGNAL("updataseg_no"), self.ncr, self.segments)

    def sortseg(self, segs):
        newsegs = []
        arrseg = np.array(segs)
        segssort = np.argsort(arrseg[:, 0])
        for ind, index in enumerate(segssort):
            newsegs.append(segs[index])
        self.segments = newsegs

    def add_items(self, row, mcrals):
        pcItem = QtGui.QTableWidgetItem(str(mcrals['pc']))
        iterItem = QtGui.QTableWidgetItem(str(mcrals['iter']))
        R2Item = QtGui.QTableWidgetItem(str(mcrals['R2']))

        pcItem.setFlags(pcItem.flags() ^ QtCore.Qt.ItemIsEditable)
        iterItem.setFlags(iterItem.flags() ^ QtCore.Qt.ItemIsEditable)
        R2Item.setFlags(R2Item.flags() ^ QtCore.Qt.ItemIsEditable)

        self.segsTable.setItem(row, 1, pcItem)
        self.segsTable.setItem(row, 2, iterItem)
        self.segsTable.setItem(row, 3, R2Item)

    def showContextMenu(self, pos):
        self.contextMenu.exec_(self.segsTable.mapToGlobal(pos))

    def showmcr(self):
        L = self.segsTable.selectedItems()
        rows1 = []
        for l in range(0, len(L)):
            rows1.append(self.segsTable.row(L[l]))
        rows = np.unique(rows1)
        if len(rows) >= 2:
            msgBox = QMessageBox()
            msgBox.setText("Only one item can be selected.")
            msgBox.exec_()
        elif len(rows) == 1:
            row = self.segsTable.row(L[0])
            seg = self.segments[row]
            tics = self.ncr.tic()
            ticval = tics['val']
            rtval = tics['rt']
            segtic = ticval[seg[0]:seg[1]]
            segrt = rtval[seg[0]:seg[1]]
            self.emit(SIGNAL("seg_no"), row)
            self.emit(SIGNAL("seg_tic"), segtic, segrt)
            if self.segsTable.item(row, 3):
                self.emit(SIGNAL("show_chrom"), self.chroms[row])
                self.emit(SIGNAL("show_mass"), self.masss[row])

    def outSelect(self, Item=None):
        if Item == None:
            return
        print(Item.text())
        row = self.segsTable.row(Item)+1
        self.emit(SIGNAL("select_row"), row)

    def undoreply(self, delete_seg):
        for index, seg in enumerate(self.segments):
            if delete_seg == seg:
                    row = index
                    break
        self.segments = self.segments[:row]+self.segments[row+1:]
        if self.segsTable.item(row, 3):
            self.chroms = self.chroms[:row]+self.chroms[row+1:]
        self.segsTable.removeRow(row)
        self.emit(SIGNAL("updataseg_no"), self.ncr, self.segments)

    def delete(self):
        L = self.segsTable.selectedItems()
        rows1 = []
        for l in range(0, len(L)):
            rows1.append(self.segsTable.row(L[l]))
        rows = np.unique(rows1)
        segs = []
        for row in rows:
            segs.append(self.segments[row])
        self.emit(SIGNAL("delete_segs"), segs)
        for row in rows:
            self.segments = self.segments[:row]+self.segments[row+1:]
            if self.segsTable.item(row, 2):
                delete_ind = np.nonzero(self.rtsor == row)
                if len(delete_ind) != 0:
                    inde = np.searchsorted(self.RT, self.rtlist[self.rtsor[delete_ind]])
                    self.RT = np.delete(self.RT, inde)
                    self.MS = np.delete(self.MS, inde, axis=0)
        for l in range(0, len(rows)):
            self.segsTable.removeRow(rows1[l]-l)
        self.emit(SIGNAL("updataseg_no"), self.ncr, self.segments)

    def updata_result(self, result, no):
        for i in range(0, self.segsTable.colorCount()):
            item = self.segsTable.item(no, i)
            item.setBackgroundColor(QColor(0, 60, 10))

        pcItem = QtGui.QTableWidgetItem(str(result['pc']))
        pcItem.setFlags(pcItem.flags() ^ QtCore.Qt.ItemIsEditable)
        self.segsTable.setItem(no, 1, pcItem)
        R2Item = QtGui.QTableWidgetItem(str(result['r2']))
        R2Item.setFlags(R2Item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.segsTable.setItem(no, 2, R2Item)

        self.rtlist.append(result['rt'])
        self.rtsor.append(no)
        if len(self.RT) == 0:
            self.MS = result['spec']
            self.RT = result['rt']
        else:
            sor = np.searchsorted(self.RT, result['rt'])
            self.MS['ms'] = np.insert(self.MSRT['ms'], sor, result['spec'], axis=0)
            self.RT['rt'] = np.insert(self.MSRT['rt'], sor, result['rt'], axis=0)


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = SEGSTable()
    window.show()
    sys.exit(app.exec_())