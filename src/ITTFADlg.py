__author__ = 'Administrator'

from MARS_methods import ittfa, fnnls
from scipy.linalg import norm

from PyQt4 import QtGui
from PyQt4.QtGui import *
import numpy as np
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class ITTFAQDialg(QWidget):
    def __init__(self, parent=None):
        super(ITTFAQDialg, self).__init__(parent)
        self.results = {}
        self.createVariabletable()
        self.create_canvas1('scan', 'Raw Scans')
        self.create_canvas2('scan', 'Resolved Chromatographic Profiles')
        self.canvas1.setMinimumWidth(800)
        self.canvas1.setMaximumWidth(800)
        self.canvas2.setMinimumWidth(800)
        self.canvas2.setMaximumWidth(800)

        ymino, ymaxo = self.axes1.get_ylim()
        xmino, xmaxo = self.axes1.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.canvas1.mpl_connect('button_press_event', self.mouse_press_callback)

        self.Addbtn = QPushButton("Add raw data")
        self.Pickbtn = QPushButton("Undo Pick")
        self.startbtn = QPushButton("Start ITTFA")
        self.undobtn = QPushButton("Undo")

        Addcomlabel = QLabel("Com:")
        self.addcomspin = QSpinBox()
        self.addcomspin.setRange(1, 10)
        Addcomlabel.setBuddy(self.addcomspin)

        ResultLabel = QLabel("R2:")
        self.R2text = QLineEdit()
        ResultLabel.setBuddy(self.R2text)

        hbbox = QHBoxLayout()
        hbbox.addWidget(ResultLabel)
        hbbox.addWidget(self.R2text)

        hbox = QGridLayout()
        hbox.addWidget(QLabel(), 0, 0)
        hbox.addWidget(self.Addbtn, 0, 1)
        hbox.addWidget(Addcomlabel, 1, 0)
        hbox.addWidget(self.addcomspin, 1, 1)
        hbox.addWidget(QLabel(), 2, 0)
        hbox.addWidget(self.Pickbtn, 2, 1)
        hbox.addWidget(QLabel(), 3, 0)
        hbox.addWidget(self.startbtn, 3, 1)
        hbox.addWidget(ResultLabel, 4, 0)
        hbox.addWidget(self.R2text, 4, 1)
        hbox.addWidget(QLabel(), 5, 0)
        hbox.addWidget(self.undobtn, 5, 1)

        hbox1 = QVBoxLayout()
        hbox1.addWidget(self.canvas1)
        hbox1.addWidget(self.canvas2)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(hbox1)
        mainLayout.addLayout(hbox)

        self.setLayout(mainLayout)
        self.resize(800, 600)
        self.move(320, 75)
        self.setWindowTitle("ITTFA")

        self.Pickbtn.clicked.connect(self.undo_mode)
        self.startbtn.clicked.connect(self.start_ittfa)
        self.undobtn.clicked.connect(self.undo)
        self.pc = 0
        self.apxs = []
        self.ms = []
        self.rt = []

    def createVariabletable(self):
        self.VariableTable = QtGui.QTableWidget(0, 1)
        self.VariableTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.VariableTable.horizontalHeader().hide()
        self.VariableTable.verticalHeader().hide()
        self.VariableTable.setShowGrid(False)
        self.VariableTable.setFixedWidth(200)

    def mouse_press_callback(self, event):
        if event.button == 3:
            self.xdata = event.xdata
            ind = np.searchsorted(range(int(self.oxy[0][0]), int(self.oxy[0][1])), event.xdata)
            self.apxs.append(ind)
            self.axes1.vlines(event.xdata, self.oxy[1][0], self.oxy[1][1],
                             color='g', linestyles='-')
            self.redraw1()

    def redraw1(self):
        self.canvas1.draw()
        self.update()

    def redraw2(self):
        self.canvas2.draw()
        self.update()

    def create_canvas1(self, xname, title):
        self.fig1 = plt.figure()
        self.axes1 = plt.subplot(111)
        self.axes1.set_xlabel(xname)
        self.axes1.set_title(title, fontsize=9)
        self.canvas1 = FigureCanvas(self.fig1)
        self.axes1.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.2, top=0.90, left=0.08, right=0.9)
        self.redraw1()

    def create_canvas2(self, xname, title):
        self.fig2 = plt.figure()
        self.axes2 = plt.subplot(111)
        self.axes2.set_xlabel(xname)
        self.axes2.set_title(title, fontsize=9)
        self.canvas2 = FigureCanvas(self.fig2)
        self.axes2.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.2, top=0.90, left=0.08, right=0.9)
        self.redraw2()

    def add_data(self, x, pc):
        self.x = x
        self.pc = pc
        self.axes1.plot(x['d'])
        self.addcomspin.setValue(pc)
        ymino, ymaxo = self.axes1.get_ylim()
        xmino, xmaxo = self.axes1.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.redraw1()

    def start_ittfa(self):
        pc = self.addcomspin.value()
        if len(self.apxs) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please Pick apex")
            msgBox.exec_()
        else:
            C = np.zeros((self.x['d'].shape[0], pc))
            for i, v in enumerate(self.apxs):
                c = ittfa(self.x['d'], v, pc)
                C[:, i] = c[:,0]
            S = np.zeros((pc, self.x['d'].shape[1]))
            for j in range(0, S.shape[1]):
                a = fnnls(np.dot(C.T, C), np.dot(C.T, self.x['d'][:, j]), tole='None')
                S[:, j] = a['xx']

            R2 = norm(np.dot(C,S))/norm(self.x['d'])

            rts = self.x['rt'][np.sort(np.argmax(C, axis=0))]
            index = np.argsort(np.argmax(C, axis=0))
            for ind, val in enumerate(index):
                self.rt.append(rts[ind])
                ss = S[val, :]
                self.ms.append(ss / norm(ss))
            self.update_fig(C)
            self.R2text.setText(str(np.round(R2, 3)))

    def undo(self):
        self.axes2.clear()
        self.axes2.set_xlabel('scan')
        self.axes2.set_title('Resolved Chromatographic Profiles', fontsize=9)
        self.axes2.tick_params(axis='both', labelsize=8)
        self.redraw2()
        self.add_data(self.x, self.pc)

    def add_items(self, result):
        self.optiterQtext.setText(str(result['itopt']))
        self.lofpcaQtext.setText(str(result['sdopt'][0]))
        self.lofexpQtext.setText(str(result['sdopt'][1]))
        self.r2Qtext.setText(str(result['r2opt']))

    def update_fig(self, C):
        self.axes2.clear()
        self.axes2.set_xlabel('scan')
        self.axes2.set_title('Resolved Chromatographic Profiles', fontsize=9)
        self.axes2.tick_params(axis='both', labelsize=8)
        self.axes2.plot(C)
        self.redraw2()

    def undo_mode(self, event):
        if len(self.apxs) >= 1:
            del self.axes1.collections[:]
            self.apxs.pop(-1)
            for ind in self.apxs:
                self.axes1.vlines(ind, self.oxy[1][0], self.oxy[1][1],
                                 color='g', linestyles='-')
            self.redraw1()
        print "undo"

    def get_resu(self):
        if len(self.rt):
            RESU = {"methods": "I", "ms": self.ms, 'rt': self.rt, 'mz': self.x['mz'], 'pc':self.pc, 'R2': 'none'}
        else:
            RESU = {}
        return RESU


if __name__ == '__main__':

    # filename = 'E:/pycharm_project/others/t1(2).cdf'
    # ncr = netcdf_reader(filename, True)
    # tic = ncr.tic()
    # mz = ncr.mz_point(100)
    # m = ncr.mat(1780, 1820, 1)
    app = QtGui.QApplication(sys.argv)
    window = ITTFAQDialg()
    # window.updata_data(m,3)
    window.show()
    sys.exit(app.exec_())