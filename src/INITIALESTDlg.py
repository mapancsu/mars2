__author__ = 'Administrator'

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from NetCDF import netcdf_reader
import numpy as np
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class INITIALESTWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(INITIALESTWidget, self).__init__(parent)

        self.createVariabletable()
        self.create_canvas()


