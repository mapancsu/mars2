__author__ = 'Administrator'

from PyQt4 import QtGui
from PyQt4.QtGui import *
import sys

class SetmarsQDialg(QDialog):
    def __init__(self, parent=None):
        super(SetmarsQDialg, self).__init__(parent)
        self.options = {'mth': "RM", 'pw': 50, 'thres': 0.8, 'w': 3, 'coef': 0.99, 'R2-PCA': 0.985,'maxCN': 10}
        self.DTZwidget = QWidget()
        self.PCOITTFAwidget = QWidget()

        self.tabWidget = QTabWidget()

        self.tabWidget.addTab(self.DTZwidget, "Detect-TZ")
        self.tabWidget.addTab(self.PCOITTFAwidget, "PCO-ITTFA")

        ## Detect-TZ widget
        MTHLabel = QLabel('Method')
        self.mthcombox = QComboBox()
        MTHLabel.setBuddy(self.mthcombox)
        self.mthcombox.addItem("RM")
        self.mthcombox.addItem("MSWFA")

        pwLabel = QLabel('peak width(scans)')
        thresLabel = QLabel('percent of max score')
        windowLabel = QLabel('Windows')
        self.pwQtext = QLineEdit()
        self.pwQtext.setEnabled(True)

        pwLabel.setBuddy(self.pwQtext)
        self.thresQspin = QDoubleSpinBox()
        self.thresQspin.setEnabled(True)
        self.thresQspin.setRange(0.80, 0.95)
        self.thresQspin.setSingleStep(0.01)

        thresLabel.setBuddy(self.thresQspin)
        self.windowQSpin = QSpinBox()
        self.windowQSpin.setEnabled(True)
        self.windowQSpin.setRange(3, 7)

        windowLabel.setBuddy(self.windowQSpin)

        layout = QGridLayout()
        layout.addWidget(MTHLabel, 0, 0)
        layout.addWidget(self.mthcombox , 0, 1)
        layout.addWidget(pwLabel, 1, 0)
        layout.addWidget(self.pwQtext, 1, 1)
        layout.addWidget(thresLabel, 2, 0)
        layout.addWidget(self.thresQspin, 2, 1)
        layout.addWidget(windowLabel, 3, 0)
        layout.addWidget(self.windowQSpin, 3, 1)

        self.DTZwidget.setLayout(layout)

        ## PCO-ITTFA widget
        coefLabel = QLabel("coefficient of SIC")
        self.thresQspin1 = QDoubleSpinBox()
        self.thresQspin1.setEnabled(True)
        self.thresQspin1.setRange(0.90, 0.99)
        self.thresQspin1.setSingleStep(0.01)
        coefLabel.setBuddy(self.thresQspin1)

        R2label = QLabel("R2-PCA")
        self.R2PCAQspin = QDoubleSpinBox()
        self.R2PCAQspin.setDecimals(3)
        self.R2PCAQspin.setEnabled(True)
        self.R2PCAQspin.setRange(0.985, 0.995)
        self.R2PCAQspin.setSingleStep(0.001)
        R2label.setBuddy(self.R2PCAQspin)

        CNlabel = QLabel("max CN")
        self.CNQspin2 = QSpinBox()
        self.CNQspin2.setEnabled(True)
        self.CNQspin2.setRange(6, 10)
        CNlabel.setBuddy(self.CNQspin2)

        layout1 = QGridLayout()
        layout1.addWidget(coefLabel, 0, 0)
        layout1.addWidget(self.thresQspin1 , 0, 1)
        layout1.addWidget(CNlabel, 1, 0)
        layout1.addWidget(self.CNQspin2, 1, 1)
        layout1.addWidget(R2label, 2, 0)
        layout1.addWidget(self.R2PCAQspin, 2, 1)
        layout1.addWidget(QLabel(), 3, 0)

        self.PCOITTFAwidget.setLayout(layout1)

        self.buttonBoxOK = QPushButton("OK")
        self.buttonBoxCancel = QPushButton("Cancel")
        vbox = QHBoxLayout()
        vbox.addStretch()
        vbox.addWidget(self.buttonBoxOK)
        vbox.addWidget(self.buttonBoxCancel)

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(self.tabWidget)
        mainlayout.addLayout(vbox)

        self.setLayout(mainlayout)
        self.setWindowTitle("Set MARS parameters")

        self.buttonBoxOK.clicked.connect(self.conditions)
        self.buttonBoxCancel.clicked.connect(self.reject)
        self.initial(self.options)

    def initial(self, opts):
        self.mthcombox.setCurrentIndex(0)
        self.pwQtext.setText(str(opts['pw']))
        self.thresQspin.setValue(opts['thres'])
        self.thresQspin1.setValue(opts['coef'])
        self.R2PCAQspin.setValue(opts['R2-PCA'])
        self.CNQspin2.setValue(opts['maxCN'])
        self.windowQSpin.setValue(opts['w'])

    def conditions(self):
        mth = self.mthcombox.currentText()
        pw = int(self.pwQtext.text())
        thres = self.thresQspin.value()
        w = self.windowQSpin.value()
        coef = self.thresQspin1.value()
        R2PCA = self.R2PCAQspin.value()
        maxCN = self.CNQspin2.value()
        self.options = {'mth': mth, 'pw': pw, 'thres': thres, 'w': w, 'coef': coef, 'R2-PCA': R2PCA, 'maxCN': maxCN}
        self.close()

    def change(self):
        if self.mthcombox.currentText()==1:
            self.windowQSpin.setEnabled(True)
        else:
            self.windowQSpin.setEnabled(False)

    def loading(self, options):
        self.options = options
        self.initial(options)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = SetmarsQDialg() #options = {'pw':50, 'thres':0.6, 'w':7}
    window.show()
    sys.exit(app.exec_())