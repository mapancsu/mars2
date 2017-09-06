__author__ = 'Administrator'


from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
import sys

class SetmarsQDialg(QDialog):
    def __init__(self, parent=None):
        super(SetmarsQDialg, self).__init__(parent)
        self.options = {'pw': 50, 'thres': 0.8, 'w': 3}
        pwLabel = QLabel('PW')
        thresLabel = QLabel('Thres')
        windowLabel = QLabel('W')
        self.pwQtext = QLineEdit()
        self.pwQtext .setEnabled(True)

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

        self.buttonBoxOK = QPushButton("OK")
        self.buttonBoxCancel = QPushButton("Cancel")
        vbox = QHBoxLayout()
        vbox.addStretch()
        vbox.addWidget(self.buttonBoxOK)
        vbox.addWidget(self.buttonBoxCancel)

        self.initial(self.options)

        mainlayout = QGridLayout()
        mainlayout.addWidget(pwLabel, 0, 0)
        mainlayout.addWidget(self.pwQtext, 0, 1)
        mainlayout.addWidget(thresLabel, 1, 0)
        mainlayout.addWidget(self.thresQspin, 1, 1)
        mainlayout.addWidget(windowLabel, 2, 0)
        mainlayout.addWidget(self.windowQSpin, 2, 1)
        mainlayout.addLayout(vbox, 3, 1)
        self.setLayout(mainlayout)

        self.buttonBoxOK.clicked.connect(self.conditions)
        self.buttonBoxCancel.clicked.connect(self.reject)

    def initial(self, opts):
        self.pwQtext.setText(str(opts['pw']))
        self.thresQspin.setValue(opts['thres'])
        self.windowQSpin.setValue(opts['w'])

    def conditions(self):
        pw = int(self.pwQtext.text())
        thres = self.thresQspin.value()
        w = self.windowQSpin.value()
        self.options = {'pw': pw, 'thres': thres, 'w': w}
        self.close()

    def loading(self, options):
        self.options = options
        self.initial(options)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = SetmarsQDialg() #options = {'pw':50, 'thres':0.6, 'w':7}
    window.show()
    sys.exit(app.exec_())