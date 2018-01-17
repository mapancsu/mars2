__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from NetCDF import netcdf_reader
import numpy as np
import sys, os

class FileFinderWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(FileFinderWidget, self).__init__(parent)

        dir = sys.path[0]+'\\data_save'

        if '\\library.zip' in dir:
            dir = dir.replace('\\library.zip', "")
        if not os.path.isdir(dir):
            os.makedirs(dir)
        self.files = {'fn':[], 'files':[],'dir': dir}

        browseButton = self.createButton("&Browse...", self.browse)
        findButton = self.createButton("&Find", self.find)
        self.fileComboBox = self.createComboBox("*.cdf")
        self.directoryComboBox = self.createComboBox(QtCore.QDir.currentPath())
        fileLabel = QtGui.QLabel("Dtp:")
        directoryLabel = QtGui.QLabel("Dir:")
        self.filesFoundLabel = QtGui.QLabel()
        self.createFilesTable()
        buttonsLayout = QtGui.QHBoxLayout()
        buttonsLayout.addStretch()
        buttonsLayout.addWidget(findButton)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(fileLabel, 0, 0)
        mainLayout.addWidget(self.fileComboBox, 0, 1, 1, 2)
        mainLayout.addWidget(directoryLabel, 1, 0)
        mainLayout.addWidget(self.directoryComboBox, 1, 1)
        mainLayout.addWidget(browseButton, 1, 2)
        mainLayout.addWidget(self.filesTable, 2, 0, 1, 3)
        mainLayout.addLayout(buttonsLayout, 3, 0, 1, 3)
        self.setLayout(mainLayout)

        self.setWindowTitle("Find Files")
        self.browse(bfirst=True)
        self.connect(self.filesTable, SIGNAL("itemDoubleClicked (QTableWidgetItem*)"), self.TICplot)

    def browse(self, bfirst=False):
        self.directoryComboBox.clear()
        if bfirst == True:
            directory1 = self.files['dir']
            directory2 = "C:\Users\Administrator\Desktop"
            directory3 = "TOF"
            self.directoryComboBox.addItem(directory1)
            self.directoryComboBox.addItem(directory2)
            self.directoryComboBox.addItem(directory3)
            self.directoryComboBox.setCurrentIndex(self.directoryComboBox.findText(directory1))
        else:
            directory = QtGui.QFileDialog.getExistingDirectory(self, "Find Files",
                    QtCore.QDir.currentPath())
            directory1 = self.files['dir']
            directory2 = "C:\Users\Administrator\Desktop"
            directory3 = "TOF"
            if directory:
                self.directoryComboBox.addItem(directory)
                self.directoryComboBox.addItem(directory1)
                self.directoryComboBox.addItem(directory2)
                self.directoryComboBox.addItem(directory3)
                self.directoryComboBox.setCurrentIndex(self.directoryComboBox.findText(directory))
            else:
                self.directoryComboBox.addItem(directory1)
                self.directoryComboBox.addItem(directory2)
                self.directoryComboBox.addItem(directory3)
                self.directoryComboBox.setCurrentIndex(self.directoryComboBox.findText(directory1))

    @staticmethod
    def updateComboBox(comboBox):
        if comboBox.findText(comboBox.currentText()) == -1:
            comboBox.addItem(comboBox.currentText())

    def find(self):
        if self.files['dir'] != self.directoryComboBox.currentText():
            self.filesTable.setRowCount(0)
        fileName = self.fileComboBox.currentText()
        path = self.directoryComboBox.currentText()
        self.updateComboBox(self.fileComboBox)
        self.updateComboBox(self.directoryComboBox)
        self.currentDir = QtCore.QDir(path)
        if not fileName:
            fileName = "*"
        files = self.currentDir.entryList([fileName],
                QtCore.QDir.Files | QtCore.QDir.NoSymLinks)
        self.showFiles(files)
        self.updata_files()

    def showFiles(self, files):
        inx = self.filesTable.rowCount()
        for i, fn in enumerate(files):
            if fn not in self.files['fn']:
                self.add_fn(fn, inx)
                inx+=1

    def add_fn(self, fn, row):
        fn1 = fn
        if '&' in fn1:
            fn1 = fn[1:]
        file = QtCore.QFile(self.currentDir.absoluteFilePath(fn1))
        size = QtCore.QFileInfo(file).size()

        fileNameItem = QtGui.QTableWidgetItem(fn)
        fileNameItem.setFlags(fileNameItem.flags() ^ QtCore.Qt.ItemIsEditable)
        sizeItem = QtGui.QTableWidgetItem("%d KB" % (int((size + 1023) / 1024)))
        sizeItem.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        sizeItem.setFlags(sizeItem.flags() ^ QtCore.Qt.ItemIsEditable)
        self.filesTable.insertRow(row)
        self.filesTable.setItem(row, 0, fileNameItem)
        self.filesTable.setItem(row, 1, sizeItem)

    def createButton(self, text, member):
        button = QtGui.QPushButton(text)
        button.clicked.connect(member)
        return button

    def createComboBox(self, text=""):
        comboBox = QtGui.QComboBox()
        comboBox.setEditable(True)
        comboBox.addItem(text)
        comboBox.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Preferred)
        return comboBox

    def createFilesTable(self):
        self.filesTable = QtGui.QTableWidget(0, 2)
        self.filesTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        self.filesTable.setHorizontalHeaderLabels(("File Name", "Size"))
        self.filesTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
        self.filesTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.filesTable.verticalHeader().hide()
        self.filesTable.setShowGrid(False)

        self.filesTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.filesTable.customContextMenuRequested.connect(self.showContextMenu)

        self.contextMenu = QtGui.QMenu(self)
        self.actionsetasref = self.contextMenu.addAction('Set as ref sample')
        self.actionTICplot = self.contextMenu.addAction('TIC plot')
        self.actiondelete = self.contextMenu.addAction('Delete')
        self.actionsetasref.triggered.connect(self.setasref)
        self.actionTICplot.triggered.connect(self.TICplot)
        self.actiondelete.triggered.connect(self.delete)

    def showContextMenu(self, pos):
        self.contextMenu.exec_(self.filesTable.mapToGlobal(pos))

    def setasref(self):
        L = self.filesTable.selectedItems()
        if len(L[0:len(L)/2]) >= 2:
             msgBox = QMessageBox()
             msgBox.setText("Only one item can be selected.")
             msgBox.exec_()
        elif len(L[0:len(L)/2]) == 1:
            if '&' in self.filesTable.item(0, 0).text():
                msgBox = QMessageBox()
                msgBox.setText("The ref has existed.")
                msgBox.exec_()
            else:
                self.filesTable.insertRow(0)
                fn = L[0].text()
                file = QtCore.QFile(self.currentDir.absoluteFilePath(fn))
                size = QtCore.QFileInfo(file).size()

                fileNameItem = QtGui.QTableWidgetItem('&' + fn)
                fileNameItem.setFlags(fileNameItem.flags() ^ QtCore.Qt.ItemIsEditable)
                sizeItem = QtGui.QTableWidgetItem("%d KB" % (int((size + 1023) / 1024)))
                sizeItem.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
                sizeItem.setFlags(sizeItem.flags() ^ QtCore.Qt.ItemIsEditable)
                self.filesTable.setItem(0, 0, fileNameItem)
                self.filesTable.setItem(0, 1, sizeItem)
                ncr = netcdf_reader(self.currentDir.absoluteFilePath(fn), bmmap=False)
                self.emit(SIGNAL("ref_ncr"), ncr)
        self.updata_files()

    def TICplot(self):
        L = self.filesTable.selectedItems()
        if len(L) != 0:
            if len(L[0:len(L)/2]) >= 2:
                msgBox = QMessageBox()
                msgBox.setText("Only one item can be selected.")
                msgBox.exec_()
            else:
                row = self.filesTable.row(L[0])
                filename = self.filesTable.item(row, 0).text()
                self.emit(SIGNAL("tic_plot"), filename, row)

    def delete(self):
        L = self.filesTable.selectedItems()
        filenames = []
        rows = self.getrows(L)
        for row in rows:
            filenames.append(self.filesTable.item(row, 0).text())
        self.emit(SIGNAL("updata_mars_results"), filenames)
        for row in rows[np.argsort(-rows)]:
            self.filesTable.removeRow(row)
        self.updata_files()

    def getrows(self, L):
        rows1 = []
        for i in range(0, len(L)):
            rows1.append(self.filesTable.row(L[i]))
        rows = np.sort(np.unique(rows1))
        return rows

    def updata_files(self):
        self.files = []
        fn = []
        files = []
        directory = self.directoryComboBox.currentText()#QtCore.QDir(self.directoryComboBox.currentText())
        rows = self.filesTable.rowCount()
        if rows > 0:
            for i in range(0, rows):
                fnl = self.filesTable.item(i, 0).text()
                fn.append(fnl)
                if '&' in fnl:
                    fnl = fnl[1:]
                files.append(self.currentDir.absoluteFilePath(fnl))
        self.files = {"fn": fn, "files": files, "dir": directory}
        self.emit(SIGNAL("update_files"), self.files)
        return self.files

    def get_files(self):
        files = self.updata_files()
        return files

    def str2Qstr(self, fns):
        dir = QString(fns['dir'])
        files = []
        fn = []
        for i in range(0, len(fns['files'])):
            fn.append(QString(fns['fn'][i]))
            files.append(QString(fns['files'][i]))
        return {'fn': fn, 'files': files, 'dir': dir}

    def loading(self, files):
        self.filesTable.setRowCount(0)
        self.filesTable.setHorizontalHeaderLabels(("File Name", "Size"))

        dir = files['dir']
        self.directoryComboBox.setEditText(QString(dir))
        path = self.directoryComboBox.currentText()
        self.updateComboBox(self.directoryComboBox)
        self.currentDir = QtCore.QDir(path)
        self.files = self.str2Qstr(files)

        for i, fn in enumerate(self.files['fn']):
            self.add_fn(fn, i)


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = FileFinderWidget()
    window.show()
    sys.exit(app.exec_())