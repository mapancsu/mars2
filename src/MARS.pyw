'''
Created on 2017.04.14

@author: Pan Ma
'''
import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from main import MainWindowWidget


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw=MainWindowWidget()
    mw.show()
    app.exec_()