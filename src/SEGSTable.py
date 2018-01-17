__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import sys

class SEGSTable(QtGui.QWidget):
    def __init__(self, parent=None):
        super(SEGSTable, self).__init__(parent)
        self.segments = []
        self.resulist = []
        self.createsegsTable()
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.segsTable)
        self.setLayout(mainLayout)
        self.connect(self.segsTable, SIGNAL("itemClicked (QTableWidgetItem*)"), self.outSelect)
        self.connect(self.segsTable, SIGNAL("itemDoubleClicked (QTableWidgetItem*)"), self.showmass)
        self.setWindowTitle("SEG")

    def createsegsTable(self):
        self.segsTable = QtGui.QTableWidget(0, 3)
        self.segsTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.segsTable.setHorizontalHeaderLabels(("SEG(min)", "COM", " MTH "))
        self.segsTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
        self.segsTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.ResizeToContents)
        self.segsTable.horizontalHeader().setResizeMode(2, QtGui.QHeaderView.ResizeToContents)
        self.segsTable.verticalHeader()
        self.segsTable.setShowGrid(True)

        self.segsTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.segsTable.customContextMenuRequested.connect(self.showContextMenu)
        self.contextMenu = QtGui.QMenu(self)
        self.actionmass = self.contextMenu.addAction('mass fig')
        self.actiondelete = self.contextMenu.addAction('Delete')
        self.actionmass.triggered.connect(self.showmass)
        self.actiondelete.triggered.connect(self.delete)

    def showContextMenu(self, pos):
        self.contextMenu.exec_(self.segsTable.mapToGlobal(pos))

    def add_segs(self, (indmin, indmax)):
        starts = []
        for seg in self.segments:
            starts.append(seg[0])
        insertrow = np.searchsorted(np.array(starts), indmin)
        seg = (indmin, indmax)
        roundseg = [round(indmin, 3), round(indmax, 3)]
        self.segments.insert(insertrow, seg)
        self.resulist.insert(insertrow, list())
        self.emit(SIGNAL("clear_massplot"))
        self.emit(SIGNAL("updataseg_no"), self.segments)
        self.add_segitem(insertrow, roundseg)

    def add_segitem(self, insertrow, roundseg):
        segItem = QtGui.QTableWidgetItem(str(roundseg))
        segItem.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        segItem.setFlags(segItem.flags() ^ QtCore.Qt.ItemIsEditable)
        self.segsTable.insertRow(insertrow)
        self.segsTable.setItem(insertrow, 0, segItem)

    def updata_resulist(self, result, seg, no):
        self.resulist[no] = result
        self.add_resuitem(result, no)

    def add_resuitem(self, result, no):
        pcItem = QtGui.QTableWidgetItem(str(result['pc']))
        pcItem.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        pcItem.setFlags(pcItem.flags() ^ QtCore.Qt.ItemIsEditable)
        self.segsTable.setItem(no, 1, pcItem)
        R2Item = QtGui.QTableWidgetItem(result['methods'])
        R2Item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        R2Item.setFlags(R2Item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.segsTable.setItem(no, 2, R2Item)
        for i in range(0, self.segsTable.columnCount()):
            item = self.segsTable.item(no, i)
            item.setTextColor(QColor(200, 111, 100))

    def outSelect(self, Item=None):
        if Item == None:
            return
        print(Item.text())
        row = self.segsTable.row(Item)+1
        self.emit(SIGNAL("select_row"), row)

    def undoreply(self, delete_seg):
        row = self.segments.index(delete_seg)
        self.segsTable.removeRow(row)
        self.resulist.pop(row)
        self.segments.pop(row)
        self.emit(SIGNAL("clear_massplot"))
        self.emit(SIGNAL("updataseg_no"), self.segments)

    def msrtlist(self):
        RT = []
        MS = []
        segno = []
        mms = []
        ssg = []
        for ind, v in enumerate(self.resulist):
            if len(v) != 0:
                rt = v['rt']
                ms = v['ms']
                mz = v['mz']
                for j in range(0, len(rt)):
                    RT.append(rt[j])
                    mms.append(ms[j])
                    ssg.append(ind+1)
        index = np.argsort(RT)
        RT = list(np.sort(RT))
        for k, val in enumerate(index):
            MS.append(mms[val])
            segno.append(ssg[val])
        return {'ms': MS, 'rt': RT, 'segno': segno, 'mz': mz}

    def showmass(self):
        Selitems = self.segsTable.selectedItems()
        rows1 = []
        for i in range(0, len(Selitems)):
            rows1.append(self.segsTable.row(Selitems[i]))
        rows = np.sort(np.unique(rows1))  # find delete rows
        if len(rows) >= 2 or len(rows) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Selected WRONG num item.")
            msgBox.exec_()
        elif self.segsTable.item(rows, 2):
            seg = self.segments[rows]
            self.emit(SIGNAL("seg_no"), rows)
            self.emit(SIGNAL("mass_plot"), self.resulist[rows], seg, rows)
        else:
            msgBox = QMessageBox()
            msgBox.setText("This is not resolved.")
            msgBox.exec_()

    def delete(self):
        Selitems = self.segsTable.selectedItems()
        rows1 = []
        for i in range(0, len(Selitems)):
            rows1.append(self.segsTable.row(Selitems[i]))
        rows = np.sort(np.unique(rows1))  # find delete rows
        segs = []
        for row in rows:
            segs.append(self.segments[row])  # get delete segments
        for row in rows[np.argsort(-rows)]:  # updata segments list and result list
            self.segsTable.removeRow(row)
            self.segments.pop(row)
            self.resulist.pop(row)
        self.emit(SIGNAL("delete_segs"), segs)  # to update segs in refplot
        # self.emit(SIGNAL("clear_massplot"))
        self.emit(SIGNAL("updataseg_no"), self.segments)  # to update segs in resoluwidget

    def renew_table(self):
        for i, res in enumerate(self.resulist):
            seg = self.segments[i]
            roundseg = [round(seg[0], 3), round(seg[1], 3)]
            self.add_segitem(i, roundseg)
            if len(res):
                self.add_resuitem(res, i)

    def clear_data(self):
        self.segsTable.clear()
        self.segsTable.setHorizontalHeaderLabels(("SEG", "COM", "MTH"))
        self.segments = []
        self.resulist = []
        self.segno = []

    def loading(self, seg, result):
        self.segments = []
        self.resulist = []
        self.segsTable.setRowCount(0)
        self.segsTable.setHorizontalHeaderLabels(("SEG", "COM", "MTH"))
        self.segments = seg
        self.resulist = result
        self.renew_table()

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = SEGSTable()
    window.show()
    sys.exit(app.exec_())