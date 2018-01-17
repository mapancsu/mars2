__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import sys

class MSRTTableWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MSRTTableWidget, self).__init__(parent)

        self.MSRT = {}
        self.finished = []
        self.createmsrtTable()

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.msrtTable)
        self.setLayout(mainLayout)
        self.setWindowTitle("MSRTTable")
        self.connect(self.msrtTable, SIGNAL("itemDoubleClicked (QTableWidgetItem*)"), self.showmasschrom)

    def createmsrtTable(self):
        self.msrtTable = QtGui.QTableWidget(0, 2)
        self.msrtTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        self.msrtTable.setHorizontalHeaderLabels(("MSRT(min)", "SEG"))
        self.msrtTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
        self.msrtTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.msrtTable.verticalHeader()
        self.msrtTable.setShowGrid(True)
        self.msrtTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.msrtTable.customContextMenuRequested.connect(self.showContextMenu)

        self.contextMenu = QtGui.QMenu(self)
        self.actionshow = self.contextMenu.addAction('Show mass')
        self.actiondelete = self.contextMenu.addAction('Delete')
        self.actionshow.triggered.connect(self.showmasschrom)
        self.actiondelete.triggered.connect(self.delete)

    def showContextMenu(self, pos):
        self.contextMenu.exec_(self.msrtTable.mapToGlobal(pos))

    def showmasschrom(self):
        Items = self.msrtTable.selectedItems()
        print(len(Items))
        if len(Items)/2 >= 2 or len(Items)==0:
            msgBox = QMessageBox()
            msgBox.setText("Only one item can be selected.")
            msgBox.exec_()
        else:
            row = self.msrtTable.row(Items[0])
            self.emit(SIGNAL("masschrom_plot"), row)

    def updatacom(self, no):
        row = no-1
        rt = self.MSRT['rt'][row]
        ms = self.MSRT['ms'][row]
        mz = self.MSRT['mz']
        self.emit(SIGNAL("masschrom_plot"), rt, ms, mz, row + 1)

    def delete(self):
        Selitems = self.msrtTable.selectedItems()
        rt = self.MSRT['rt']
        ms = self.MSRT['ms']
        segno = self.MSRT['segno']
        mz = self.MSRT['mz']
        if len(Selitems) >= 2:
            rows1 = []
            for i in range(0, len(Selitems)):
                rows1.append(self.msrtTable.row(Selitems[i]))
            rows = []
            for row in np.sort(np.unique(rows1))[::-1]:
                if row in self.finished:
                    ro = np.searchsorted(self.finished, row)
                    # self.finished.pop(ro)
                    rows.append(ro)
                self.msrtTable.removeRow(row)
                rt.pop(row)
                ms.pop(row)
                segno.pop(row)
                self.MSRT = {'ms': ms, 'rt': rt, 'segno': segno, 'mz': mz}
            self.emit(SIGNAL("updata_msrt"), self.MSRT, rows)

    def add_msrt(self, msrt):
        if self.MSRT != msrt['rt']:
            rts = msrt['rt']
            fin = []
            for i in self.finished:
                row = np.searchsorted(rts, self.MSRT['rt'][i])
                fin.append(row)
            self.finished = fin
            self.MSRT = msrt
            self.add_rtitem(msrt)

    def add_rtitem(self, msrt):
        self.msrtTable.clear()
        self.msrtTable.setHorizontalHeaderLabels(("MSRT(min)", "SEG"))
        rt = msrt['rt']
        segno = msrt['segno']

        for ind, val in enumerate(rt):
            rtItem = QtGui.QTableWidgetItem(str(np.round(val, 3)))
            rtItem.setFlags(rtItem.flags() ^ QtCore.Qt.ItemIsEditable)
            rtItem.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignVCenter)
            if self.msrtTable.rowCount() < len(rt):
                self.msrtTable.insertRow(ind)
            self.msrtTable.setItem(ind, 0, rtItem)

            segnoItem = QtGui.QTableWidgetItem(str(segno[ind]))
            segnoItem.setFlags(segnoItem.flags() ^ QtCore.Qt.ItemIsEditable)
            segnoItem.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.msrtTable.setItem(ind, 1, segnoItem)

            if ind in self.finished:
                item0 = self.msrtTable.item(ind, 0)
                item0.setTextColor(QColor(200, 111, 100))
                item1 = self.msrtTable.item(ind, 1)
                item1.setTextColor(QColor(200, 111, 100))

    def update_resulist(self, resulist):
        self.resulist = resulist
        if len(resulist):
            tt = len(resulist[0]['areas'])
            self.finished = range(0, len(resulist[0]['areas']))
            self.add_rtitem(self.MSRT)
        else:
            self.finished = []

    def get_MSRT(self):
        return self.MSRT

    def get_rtitem(self):
        rows = self.msrtTable.rowCount()
        rtitems = []
        for i in range(0, rows):
            if self.msrtTable.item(i, 0):
                rtitems.append(str(self.msrtTable.item(i, 0).text()))
        return rtitems

    def clear_data(self):
        self.MSRT = {}
        self.msrtTable.setRowCount(0)
        self.finished = []
        self.msrtTable.setHorizontalHeaderLabels(("MSRT(min)", "SEG"))

    def loading(self, msrt, finished):
        self.clear_data()
        self.MSRT = msrt
        self.finished = finished
        if len(msrt):
            self.add_rtitem(msrt)



if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MSRTTableWidget()
    window.show()
    sys.exit(app.exec_())