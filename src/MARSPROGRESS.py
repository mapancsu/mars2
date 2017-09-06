__author__ = 'Administrator'
import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
import numpy as np
import pickle

class MCRALSQDialg(QProgressDialog):
    def __init__(self, fileno, parent=None):
        super(MCRALSQDialg, self).__init__(parent)

        numLabel=QLabel("files:")
        self.numLineEdit = QLineEdit(str(fileno))
        typeLabel=QLabel("method:")
        self.methods = QLineEdit('mars')
        # self.typeComboBox=QComboBox()
        # self.typeComboBox.addItem('case1')
        # self.typeComboBox.addItem('case2')

        self.progressBar = QProgressBar()

        layout = QGridLayout()
        layout.addWidget(numLabel,0,0)
        layout.addWidget(self.numLineEdit,0,1)
        layout.addWidget(typeLabel,1,0)
        layout.addWidget(self.methods,1,1)
        layout.addWidget(self.progressBar,2,1)
        # layout.setMargin(15)
        # layout.setSpacing(10)

        self.setLayout(layout)
        self.setWindowTitle('mars')

        self.slotStart()

    def slotStart(self):
        num = int(self.numLineEdit.text())

        # if self.typeComboBox.currentIndex()== 0:
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(num)

        for i in range(num):
            self.progressBar.setValue(i)
            QThread.msleep(100)

        # elif self.typeComboBox.currentIndex()==1:
        #     progressDialog=QProgressDialog(self)
        #     progressDialog.setWindowModality(Qt.WindowModal)
        #     progressDialog.setMinimumDuration(5)
        #     progressDialog.setWindowTitle('wait')
        #     progressDialog.setLabelText('copy')
        #     progressDialog.setCancelButtonText('cancle')
        #     progressDialog.setRange(0,num)
        #
        #     for i in range(num):
        #         progressDialog.setValue(i)
        #         QThread.msleep(100)
        #         if progressDialog.wasCanceled():
        #             return


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = MCRALSQDialg(6)
    window.show()
    sys.exit(app.exec_())