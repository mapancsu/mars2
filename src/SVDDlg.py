__author__ = 'Administrator'

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector, Button
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from NetCDF import netcdf_reader
from chemoMethods import pcarep


class SVDQDialg(QWidget):
    def __init__(self, parent=None):
        super(SVDQDialg, self).__init__(parent)

        self.createeigentable()
        self.create_canvas()
        self.canvas.setFixedWidth(825)
        self.pc_numbers = 0

        SVDLabel = QLabel("Sigular Value Decomposition")
        # SVDLabel.setFont(QFont().setPointSize(12))

        ComsLabel = QLabel("Selected Number of Components:")
        self.numberLineEdit = QLineEdit()
        self.numberLineEdit.setEnabled(False)
        self.numberLineEdit.setText(str(1))
        self.numberLineEdit.setStyleSheet("color:red")
        ComsLabel.setBuddy(self.numberLineEdit)

        self.starbutton = QPushButton("start")
        self.starbutton.clicked.connect(self.do_svd)

        # self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # self.buttonBox.button(QDialogButtonBox.Ok).setDefault(True)
        # self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        # self.connect(self.buttonBox, SIGNAL("rejected()"), self.close)
        # self.connect(self.buttonBox, SIGNAL("accepted()"), self.accept)

        HorizonLayout0 = QHBoxLayout()
        HorizonLayout0.addStretch()
        HorizonLayout0.addWidget(SVDLabel)
        HorizonLayout0.addStretch()

        HorizonLayout1 = QHBoxLayout()
        HorizonLayout1.addWidget(self.eigensTable)
        HorizonLayout1.addWidget(self.canvas)

        HorizonLayout2 = QHBoxLayout()
        HorizonLayout2.addStretch()
        HorizonLayout2.addWidget(ComsLabel)
        HorizonLayout2.addWidget(self.numberLineEdit)
        HorizonLayout2.addStretch()
        HorizonLayout2.addWidget(self.starbutton)

        VerticalLayout = QVBoxLayout()
        VerticalLayout.addLayout(HorizonLayout0)
        VerticalLayout.addLayout(HorizonLayout1)
        VerticalLayout.addLayout(HorizonLayout2)

        self.setLayout(VerticalLayout)
        self.resize(1000, 600)
        self.move(320, 75)
        self.connect(self.eigensTable, SIGNAL("itemClicked (QTableWidgetItem*)"), self.outSelect)
        self.starbutton.clicked.connect(self.do_svd)
        # self.connect(self.eigensTable, SIGNAL('itemClicked(int, int)'), self.updata_canvas)
        # self.connect(self.eigensTable, SIGNAL(cellClicked(int, int)), this, SLOT(myCellClicked(int, int)))
        # self.eigensTable.itemClicked.connect(self.updata_canvas)

    def createeigentable(self):
        self.eigensTable = QtGui.QTableWidget(0, 1)
        # self.eigensTable.setWindowTitle("EigensValue")
        self.eigensTable.setHorizontalHeaderLabels(("EigensValue",))
        self.eigensTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
        self.eigensTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.eigensTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.eigensTable.verticalHeader().hide()
        self.eigensTable.setShowGrid(False)
        # self.eigensTable.setEditTriggers(QtGui.QAbstractItemView.SelectedClicked)

    def redraw(self):
        self.canvas.draw()
        self.update()

    def create_canvas(self):
        self.fig = plt.figure()
        self.axes1 = plt.subplot(311)
        self.axes2 = plt.subplot(312)
        self.axes3 = plt.subplot(313)
        self.fig.subplots_adjust(wspace=0.35, hspace=0.35)
        # self.axes1.set_xlabel("Scans no")
        # self.axes1.set_ylabel("Intensity")
        # self.axes2.set_xlabel("Scans no")
        # self.axes3.set_xlabel("Scans no")
        # #plt.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.fig)

        self.axes1.set_title("Raw data matrix", fontsize=9)
        self.axes2.set_title("EigenValues Reprezentation", fontsize=9)
        self.axes3.set_title("EigenVectors Reprezentation", fontsize=9)
        self.axes1.tick_params(axis='both', labelsize=8)
        self.axes2.tick_params(axis='both', labelsize=8)
        self.axes3.tick_params(axis='both', labelsize=8)
        self.redraw()

    def outSelect(self, Item=None):
        #self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        self.axes3.clear()
        if Item == None:
            return
        row = self.eigensTable.row(Item)+1
        self.numberLineEdit.setText(str(row))
        del self.axes3.collections[:]
        self.axes3.plot(self.U[:, 0:row])
        self.redraw()

    def updata_data(self, x):
        self.x = x
        # U, D, V = np.linalg.svd(np.dot(self.x['d'], self.x['d'].T))
        # self.U = U
        # self.D = D
        # self.eigs = np.power(D, 0.5)
        #
        # self.axes1.plot(self.x['d'])
        # self.axes1.set_ylim(np.min(self.x['d']), np.max(self.x['d'])*1.1)
        # self.axes2.scatter(range(1, U.shape[0]+1), self.eigs, s=80, marker='o')
        # # self.axes2.plot(self.eigs, s=20, c='b', marker='o')
        # self.axes2.set_ylim(-0.1*np.max(self.eigs), np.max(self.eigs)*1.1)
        # self.axes2.set_xlim(-1, len(self.eigs))
        # self.axes3.plot(self.U[:, 0])
        #
        # # self.axes1.set_xlim(np.min(self.x), np.max(self.x))
        # # self.axes1.set_ylim(0, len(self.eigs))
        #
        # for eig in self.eigs:
        #     eigNameItem = QtGui.QTableWidgetItem(str(eig))
        #     eigNameItem.setFlags(eigNameItem.flags() ^ QtCore.Qt.ItemIsEditable)
        #     row = self.eigensTable.rowCount()
        #     self.eigensTable.insertRow(row)
        #     self.eigensTable.setItem(row, 0, eigNameItem)

    def do_svd(self):
        U, D, V = np.linalg.svd(np.dot(self.x['d'], self.x['d'].T))
        self.U = U
        self.D = D
        self.eigs = np.power(D, 0.5)

        self.axes1.plot(self.x['d'])
        self.axes1.set_ylim(np.min(self.x['d']), np.max(self.x['d'])*1.1)
        self.axes2.scatter(range(1, U.shape[0]+1), self.eigs, s=80, marker='o')
        # self.axes2.plot(self.eigs, s=20, c='b', marker='o')
        self.axes2.set_ylim(-0.1*np.max(self.eigs), np.max(self.eigs)*1.1)
        self.axes2.set_xlim(-1, len(self.eigs))
        self.axes3.plot(self.U[:, 0])

        # self.axes1.set_xlim(np.min(self.x), np.max(self.x))
        # self.axes1.set_ylim(0, len(self.eigs))

        for eig in self.eigs:
            eigNameItem = QtGui.QTableWidgetItem(str(eig))
            eigNameItem.setFlags(eigNameItem.flags() ^ QtCore.Qt.ItemIsEditable)
            row = self.eigensTable.rowCount()
            self.eigensTable.insertRow(row)
            self.eigensTable.setItem(row, 0, eigNameItem)


    def accept(self):
        self.pc_numbers = int(self.numberLineEdit.text())
        self.close()
        # self.emit(SIGNAL('pc_numbers'), self.pc_numbers)

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = SVDQDialg()
    window.show()
    sys.exit(app.exec_())

