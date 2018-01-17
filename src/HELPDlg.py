__author__ = 'Administrator'


from scipy.linalg import norm
from matplotlib.widgets import SpanSelector

from HELP import FR, FSWFA,svdX
from MARS_methods import fnnls

from PyQt4 import QtGui
from PyQt4.QtGui import *
import numpy as np
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pickle

class HELPQDialg(QWidget):
    def __init__(self, parent=None):
        super(HELPQDialg, self).__init__(parent)
        self.results = {}
        self.help_result = []
        self.pc = 0
        self.ms = []
        self.rt = []
        self.createVariabletable()
        self.create_canvas1('scan', 'ETA or LPG')
        self.create_canvas2('scan', 'Resolved Chromatographic Profiles')
        self.canvas1.setMinimumWidth(800)
        self.canvas1.setMaximumWidth(800)
        self.canvas2.setMinimumWidth(800)
        self.canvas2.setMaximumWidth(800)

        self.mthcombox = QComboBox()
        self.mthcombox.addItem("ETA")
        self.mthcombox.addItem("LPG")
        self.DoETA = QPushButton("Do")

        LAYER = QLabel("LAYER:")
        self.LComboBox = QSpinBox()

        wLabel = QLabel("W:")
        self.DComboBox = QSpinBox()
        self.DComboBox.setRange(3, 9)
        self.DComboBox.setSingleStep(2)
        self.DComboBox.setValue(7)

        CLabel1 = QLabel("~")
        CLabel2 = QLabel("~")
        CLabel3 = QLabel("~")
        Selabel = QLabel("SE:")
        self.SComboBox1 = QLineEdit()
        self.SComboBox2 = QLineEdit()
        self.sbtn = QPushButton("...")

        Ovlabel = QLabel("OV:")
        self.obtn1 = QPushButton("...")
        self.OComboBox1 = QLineEdit()
        self.OComboBox2 = QLineEdit()

        self.OComboBox3 = QLineEdit()
        self.OComboBox4 = QLineEdit()

        self.obtn2 = QPushButton("...")

        self.DoFR = QPushButton("FR")

        self.addbtn = QPushButton("Add raw data")

        self.undobtn = QPushButton("Undo")

        hbox0 = QHBoxLayout()
        hbox0.addWidget(LAYER)
        hbox0.addWidget(self.LComboBox)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(wLabel)
        hbox1.addWidget(self.DComboBox)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.mthcombox)
        hbox2.addWidget(self.DoETA)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(Selabel)
        hbox3.addWidget(self.SComboBox1)
        hbox3.addWidget(CLabel1)
        hbox3.addWidget(self.SComboBox2)
        hbox3.addWidget(self.sbtn)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(Ovlabel)
        hbox4.addWidget(self.OComboBox1)
        hbox4.addWidget(CLabel2)
        hbox4.addWidget(self.OComboBox2)
        hbox4.addWidget(self.obtn1)

        hbox5 = QHBoxLayout()
        hbox5.addWidget(Ovlabel)
        hbox5.addWidget(self.OComboBox3)
        hbox5.addWidget(CLabel3)
        hbox5.addWidget(self.OComboBox4)
        hbox5.addWidget(self.obtn2)

        hbox6 = QHBoxLayout()
        hbox6.addStretch()
        hbox6.addWidget(self.DoFR)

        hbox7 = QHBoxLayout()
        hbox7.addStretch()
        hbox7.addWidget(self.undobtn)

        GLay = QGridLayout()
        GLay.addWidget(self.addbtn, 0, 0)
        GLay.addLayout(hbox0, 1, 0)
        GLay.addLayout(hbox1, 2, 0)
        GLay.addLayout(hbox2, 3, 0)
        GLay.addLayout(hbox3, 4, 0)
        GLay.addLayout(hbox4, 5, 0)
        GLay.addLayout(hbox5, 6, 0)
        GLay.addLayout(hbox6, 7, 0)
        GLay.addLayout(hbox7, 8, 0)

        VLay = QVBoxLayout()
        VLay.addWidget(self.canvas1)
        VLay.addWidget(self.canvas2)

        VLay1 = QVBoxLayout()
        VLay1.addLayout(GLay)
        VLay1.addStretch()

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(VLay)
        mainLayout.addLayout(VLay1)

        self.setLayout(mainLayout)
        self.resize(800, 600)
        self.move(320, 75)
        self.setWindowTitle("HELP")

        self.span1.set_active(True)
        self.span2.set_active(False)
        self.span3.set_active(False)

        self.DoETA.clicked.connect(self.eta)
        self.DoFR.clicked.connect(self.fr)
        self.undobtn.clicked.connect(self.undo)

        self.sbtn.clicked.connect(self.span_mode1)
        self.obtn1.clicked.connect(self.span_mode2)
        self.obtn2.clicked.connect(self.span_mode3)

    def createVariabletable(self):
        self.VariableTable = QtGui.QTableWidget(0, 1)
        self.VariableTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.VariableTable.horizontalHeader().hide()
        self.VariableTable.verticalHeader().hide()
        self.VariableTable.setShowGrid(False)
        self.VariableTable.setFixedWidth(200)

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
        ymino, ymaxo = self.axes1.get_ylim()
        xmino, xmaxo = self.axes1.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]

        self.span1 = SpanSelector(self.axes1, self.span_select_callback1, 'horizontal', minspan=0.002, useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'),
                                 onmove_callback=None,
                                 button=[1])
        self.span2 = SpanSelector(self.axes1, self.span_select_callback2, 'horizontal', minspan=0.002, useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='green'),
                                 onmove_callback=None,
                                 button=[1])
        self.span3 = SpanSelector(self.axes1, self.span_select_callback3, 'horizontal', minspan=0.002, useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='green'),
                                 onmove_callback=None,
                                 button=[1])
        plt.subplots_adjust(bottom=0.2, top=0.90, left=0.08, right=0.9)
        self.redraw1()
        self.span1.set_active(True)

    def create_canvas2(self, xname, title):
        self.fig2 = plt.figure()
        self.axes2 = plt.subplot(111)
        self.axes2.set_xlabel(xname)
        self.axes2.set_title(title, fontsize=9)
        self.canvas2 = FigureCanvas(self.fig2)
        self.axes2.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.2, top=0.90, left=0.08, right=0.9)
        self.redraw2()

    def span_mode1(self, event):
        self.span1.set_active(True)
        self.span2.set_active(False)
        self.span3.set_active(False)
        self.span1.connect_event('motion_notify_event', self.span1.onmove)
        self.span1.connect_event('button_press_event', self.span1.press)
        self.span1.connect_event('button_release_event', self.span1.release)
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw1()
        print "span1"

    def span_mode2(self, event):
        self.span1.set_active(False)
        self.span3.set_active(False)
        self.span2.set_active(True)
        self.span2.connect_event('motion_notify_event', self.span2.onmove)
        self.span2.connect_event('button_press_event', self.span2.press)
        self.span2.connect_event('button_release_event', self.span2.release)
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw1()
        print "span2"

    def span_mode3(self, event):
        self.span1.set_active(False)
        self.span2.set_active(False)
        self.span3.set_active(True)
        self.span3.connect_event('motion_notify_event', self.span3.onmove)
        self.span3.connect_event('button_press_event', self.span3.press)
        self.span3.connect_event('button_release_event', self.span3.release)
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw1()
        print "span3"

    def span_select_callback1(self, xmin, xmax):
        ax = np.arange(0, self.x['d'].shape[0])
        imin, imax = np.searchsorted(ax, (xmin, xmax))
        imax = min(len(ax)-1, imax)
        seg = ax[[imin, imax]]
        # self.axes1.scatter(xmin, self.y[imin], s=30, marker='^',
        #     color='red', label='x1 samples')
        # self.axes1.scatter(xmax, self.y[imax], s=30, marker='v',
        #     color='red', label='x1 samples')
        self.axes1.fill_between(range(imin, imax), self.oxy[1][0], self.oxy[1][1], facecolor='yellow', alpha=0.5)
        self.redraw1()
        self.SComboBox1.setText(str(imin))
        self.SComboBox2.setText(str(imax))

    def span_select_callback2(self, xmin, xmax):
        ax = np.arange(0, self.x['d'].shape[0])
        imin, imax = np.searchsorted(ax, (xmin, xmax))
        imax = min(len(ax)-1, imax)
        seg = ax[[imin, imax]]
        # self.axes1.scatter(xmin, self.y[imin], s=30, marker='^',
        #     color='red', label='x1 samples')
        # self.axes1.scatter(xmax, self.y[imax], s=30, marker='v',
        #     color='red', label='x1 samples')
        self.axes1.fill_between(range(imin, imax), self.oxy[1][0], self.oxy[1][1], facecolor='Green', alpha=0.5)
        self.redraw1()
        self.OComboBox1.setText(str(imin))
        self.OComboBox2.setText(str(imax))

    def span_select_callback3(self, xmin, xmax):
        ax = np.arange(0, self.x['d'].shape[0])
        imin, imax = np.searchsorted(ax, (xmin, xmax))
        imax = min(len(ax)-1, imax)
        seg = ax[[imin, imax]]
        self.axes1.fill_between(range(imin,imax), self.oxy[1][0], self.oxy[1][1], facecolor='Green', alpha=0.5)
        self.redraw1()
        self.OComboBox3.setText(str(imin))
        self.OComboBox4.setText(str(imax))

    def add_data(self, x, pc):
        self.x = x
        self.pc = pc
        self.new_x = x['d']
        self.axes1.plot(x['d'])
        self.axes1.set_title("raw data", fontsize=9)
        self.axes1.set_xlabel("Scans")
        self.axes1.tick_params(axis='both', labelsize=8)
        self.LComboBox.setRange(1, self.pc)
        ymino, ymaxo = self.axes1.get_ylim()
        xmino, xmaxo = self.axes1.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.redraw1()

    def eta(self):
        self.layel = self.LComboBox.value()
        if self.layel < self.pc:
            if self.mthcombox.currentText() == "ETA":
                w = self.DComboBox.value()
                self.l, self.em = FSWFA(self.new_x, w, self.pc)
                self.axes1.clear()
                self.axes1.plot(self.l, self.em, '-o')
                self.axes1.set_title("ETA", fontsize=9)
                self.axes1.set_xlabel("Scans")
                self.axes1.tick_params(axis='both', labelsize=8)
                self.redraw1()
            else:
                u, T = svdX(self.new_x)
                self.axes1.clear()
                self.axes1.plot(u[:, 0], u[:, 1], '-o')
                C = np.arange(0, u.shape[0])
                for a, b, c in zip(u[:, 0], u[:, 1], C):
                    self.axes1.text(a, b + 0.001, '%.0f' % c, ha='center', va='bottom', fontsize=7)
                self.axes1.set_xlabel("p1")
                self.axes1.set_ylabel("p2")
                self.axes1.set_title("LPG", fontsize=9)
                self.redraw1()
            ymino, ymaxo = self.axes1.get_ylim()
            xmino, xmaxo = self.axes1.get_xlim()
            self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        else:
            msgBox = QMessageBox()
            msgBox.setText("All component finished")
            msgBox.exec_()

    def fr(self):
        s1 = self.SComboBox1.text()
        s2 = self.SComboBox2.text()
        z1 = self.OComboBox1.text()
        z2 = self.OComboBox2.text()
        z3 = self.OComboBox3.text()
        z4 = self.OComboBox4.text()
        s = list()
        z = list()
        if (s1 and s2):
            s.append(range(int(s1), int(s2)))
        if (z1 and z2):
            z.append(range(int(z1), int(z2)))
        if (z3 and z4):
            z.append(range(int(z3), int(z4)))
        if len(s[0]) == 0 or len(z[0]) == 0:
            msgBox = QMessageBox()
            msgBox.setText("Please input selective region or zero-concentration region")
            msgBox.exec_()
        if len(s) and len(z):
            DATAF = open('HELP_m.pkl', 'w')
            X = {'x': self.new_x, 'so':s[0], 'z':z[0]}
            pickle.dump(X, DATAF)
            DATAF.close()
            c, new_x = FR(self.new_x, s[0], z[0], self.pc-self.LComboBox.value()+1)
            # plt.plot(c)
            # plt.show()
            # plt.plot(new_x)
            # plt.show()

            helpre = {'new_x': self.new_x, 'l': self.l, 'em': self.em, 'c': c}
            if len(self.help_result) == self.layel:
                self.help_result.pop(self.layel)
            self.help_result.append(helpre)

            self.new_x = new_x
            self.LComboBox.setValue(self.LComboBox.value()+1)
            self.SComboBox1.setText(str())
            self.SComboBox2.setText(str())
            self.OComboBox1.setText(str())
            self.OComboBox2.setText(str())
            self.OComboBox3.setText(str())
            self.OComboBox4.setText(str())

            self.axes2.plot(c/np.linalg.norm(c))
            self.redraw2()

            self.axes1.clear()
            self.axes1.set_xlabel("scan")
            self.axes1.plot(self.new_x)
            self.axes1.set_title("after stripping", fontsize=9)
            self.axes1.set_xlabel("Scans")
            self.redraw1()
            ymino, ymaxo = self.axes1.get_ylim()
            xmino, xmaxo = self.axes1.get_xlim()
            self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]

            if self.layel == self.pc-1:
                #cc = np.sum(self.new_x, 1)
                indd = np.argmax(np.sum(self.new_x, 0))
                cc = self.new_x[:,indd]
                ind = np.argmax(cc)

                for i, indd in enumerate(np.arange(ind, 0, -1)):
                    if cc[indd-1] >= cc[indd] and cc[indd-1] <= 0.5*np.max(cc):
                        cc[0:indd] = 0
                        break
                    if cc[indd-1] < 0:
                        cc[0:indd] = 0
                        break

                for i, indd in enumerate(np.arange(ind, len(cc)-1, 1)):
                    if cc[indd+1] >= cc[indd] and cc[indd+1] <= 0.5*np.max(cc):
                        cc[indd+1:len(cc)] = 0
                        break
                    if cc[indd+1] < 0:
                        cc[indd+1:len(cc)] = 0
                        break

                helpre = {'new_x': [], 'l': [], 'em': [], 'c': cc/np.linalg.norm(cc)}
                self.help_result.append(helpre)
                self.axes2.plot(cc/np.linalg.norm(cc))
                self.redraw2()
                return

            self.layel = self.layel+1
            w = self.DComboBox.value()
            self.l, self.em = FSWFA(self.new_x, w, self.pc-self.LComboBox.value()+1)

    def undo(self):
        self.axes1.clear()
        self.axes1.set_xlabel("scan")
        self.axes2.clear()
        self.axes2.set_xlabel("scan")
        self.LComboBox.setValue(1)
        self.help_result = []
        self.rt = []
        self.ms = []
        self.new_x = self.x['d']
        self.axes1.plot(self.new_x)
        self.axes1.set_title("raw data", fontsize=9)
        self.axes1.set_xlabel("Scans")
        self.axes1.tick_params(axis='both', labelsize=8)
        self.redraw2()
        self.redraw1()

    def getmsrt(self):
        if len(self.help_result) == self.pc:
            C = np.zeros((self.x['d'].shape[0], self.pc))
            for i, v in enumerate(self.help_result):
                C[:, i] = self.help_result[i]['c']
            S = np.zeros((self.pc, self.x['d'].shape[1]))
            for j in range(0, S.shape[1]):
                a = fnnls(np.dot(C.T, C), np.dot(C.T, self.x['d'][:, j]), tole='None')
                S[:, j] = a['xx']

            rts = self.x['rt'][np.sort(np.argmax(C, axis=0))]
            index = np.argsort(np.argmax(C, axis=0))
            for ind, val in enumerate(index):
                self.rt.append(rts[ind])
                ss = S[val, :]
                self.ms.append(ss / norm(ss))
        else:
            msgBox = QMessageBox()
            msgBox.setText("Please resolve all components")
            msgBox.exec_()

    def get_resu(self):
        self.getmsrt()
        if len(self.rt):
            RESU = {"methods": "H", "ms": self.ms, 'rt': self.rt, 'mz': self.x['mz'], 'pc':self.pc, 'R2': 'none'}
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
    window = HELPQDialg()
    # window.updata_data(m,3)
    window.show()
    sys.exit(app.exec_())