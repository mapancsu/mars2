__author__ = 'Administrator'

from scipy.linalg import norm
from chemoMethods import mcr_als
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
import numpy as np
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from chemoMethods import pcarep, pure

class MCRALSQDialg(QWidget):
    def __init__(self, parent=None):
        super(MCRALSQDialg, self).__init__(parent)
        self.results = {}
        self.init = 0
        self.ms = []
        self.rt = []
        self.r2opt = []
        self.createVariabletable()
        self.create_canvas1('scan', 'Purest variable')
        self.create_canvas2('scan', 'Resolved Chromatographic Profiles')
        self.canvas1.setMinimumWidth(800)
        self.canvas1.setMaximumWidth(800)
        self.canvas2.setMinimumWidth(800)
        self.canvas2.setMaximumWidth(800)

        purelistLabel = QLabel("Purest variable")
        DirectonLabel = QLabel("Direction:")
        self.DComboBox = QComboBox()
        self.DComboBox.addItem("Concentration")
        self.DComboBox.addItem("Spectrum")
        self.DComboBox.setCurrentIndex(1)
        DirectonLabel.setBuddy(self.DComboBox)
        NoiseLabel = QLabel("Noise level:")
        self.NLinedit = QLineEdit()
        self.NLinedit.setText(str(10))
        self.NLinedit.setStyleSheet("color:red")
        NoiseLabel.setBuddy(self.NLinedit)
        self.DoButton = QPushButton("Start1")
        vbox = QVBoxLayout()
        vbox.addWidget(purelistLabel)
        vbox.addWidget(self.VariableTable)
        gridbox = QGridLayout()
        gridbox.addWidget(DirectonLabel, 0, 0)
        gridbox.addWidget(self.DComboBox, 0, 1)
        gridbox.addWidget(NoiseLabel, 1, 0)
        gridbox.addWidget(self.NLinedit, 1, 1)
        gridbox.addWidget(self.DoButton, 2, 1)
        vvbox = QVBoxLayout()
        vvbox.addLayout(vbox)
        vvbox.addStretch()
        vvbox.addLayout(gridbox)
        purelayout = QHBoxLayout()
        purelayout.addWidget(self.canvas1)
        purelayout.addLayout(vvbox)

        optiterLabel = QLabel("ITER_OPT:")
        self.optiterQtext = QLineEdit()
        self.optiterQtext.setEnabled(False)
        lofpcaLaber = QLabel("LOF_PCA:")
        self.lofpcaQtext = QLineEdit()
        self.lofpcaQtext.setEnabled(False)
        lofexpLaber = QLabel("LOF_EXP:")
        self.lofexpQtext = QLineEdit()
        self.lofexpQtext.setEnabled(False)
        r2Laber = QLabel("R2:")
        self.r2Qtext = QLineEdit()
        self.r2Qtext.setEnabled(False)
        nonnegativeLabel1 = QLabel("CHROM NNEG:")
        self.nonnegativeBox1 = QComboBox()
        self.nonnegativeBox1.addItem("set to 0")
        self.nonnegativeBox1.addItem("nnls")
        self.nonnegativeBox1.addItem("fnnls")
        self.nonnegativeBox1.setCurrentIndex(2)
        nonnegativeLabel1.setBuddy(self.nonnegativeBox1)

        unimodalityLabel = QLabel("CHROM UNIMOD:")
        self.unimodalityBox = QComboBox()
        self.unimodalityBox.addItem("horizon")
        self.unimodalityBox.addItem("vetical")
        self.unimodalityBox.addItem("average")
        self.unimodalityBox.setCurrentIndex(2)
        unimodalityLabel.setBuddy(self.unimodalityBox)

        nonnegativeLabel2 = QLabel("MASS NNEG:")
        self.nonnegativeBox2 = QComboBox()
        self.nonnegativeBox2.addItem("set to 0")
        self.nonnegativeBox2.addItem("nnls")
        self.nonnegativeBox2.addItem("fnnls")
        self.nonnegativeBox2.setCurrentIndex(2)
        nonnegativeLabel2.setBuddy(self.nonnegativeBox2)
        self.startbutton = QPushButton("Start2")

        gridbox0 = QGridLayout()
        gridbox0.addWidget(optiterLabel, 0, 0)
        gridbox0.addWidget(self.optiterQtext, 0, 1)
        gridbox0.addWidget(lofpcaLaber, 1, 0)
        gridbox0.addWidget(self.lofpcaQtext, 1, 1)
        gridbox0.addWidget(lofexpLaber, 2, 0)
        gridbox0.addWidget(self.lofexpQtext, 2, 1)
        gridbox0.addWidget(r2Laber, 3, 0)
        gridbox0.addWidget(self.r2Qtext, 3, 1)

        gridbox1 = QGridLayout()
        gridbox1.addWidget(nonnegativeLabel1, 0, 0)
        gridbox1.addWidget(self.nonnegativeBox1, 0, 1)
        gridbox1.addWidget(unimodalityLabel, 1, 0)
        gridbox1.addWidget(self.unimodalityBox, 1, 1)
        gridbox1.addWidget(nonnegativeLabel2, 2, 0)
        gridbox1.addWidget(self.nonnegativeBox2, 2, 1)
        gridbox1.addWidget(self.startbutton, 3, 1)
        vbox = QVBoxLayout()
        vbox.addLayout(gridbox0)
        vbox.addStretch()
        vbox.addLayout(gridbox1)

        ITERlayout = QHBoxLayout()
        ITERlayout.addWidget(self.canvas2)
        ITERlayout.addLayout(vbox)

        # self.buttonBoxOK = QPushButton("OK")
        # self.buttonBoxOK.setEnabled(False)
        # self.buttonBoxCancel = QPushButton("Cancel")

        hbox = QHBoxLayout()
        hbox.addStretch()
        # hbox.addWidget(self.buttonBoxOK)
        # hbox.addWidget(self.buttonBoxCancel)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(purelayout)
        mainLayout.addLayout(ITERlayout)
        mainLayout.addLayout(hbox)
        self.setLayout(mainLayout)
        self.resize(800, 600)
        self.move(320, 75)
        self.setWindowTitle("MCR-ALS")

        # self.DoButton.clicked.connect(self.puremethod)
        # self.startbutton.clicked.connect(self.mcrals)
        # self.buttonBoxOK.clicked.connect(self.accept)

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

    def updata_data(self, x, pc):
        self.X = x
        self.x = x['d']
        self.pc = pc
        self.rttic = x['rt']
        u, s, v, d, sigma = pcarep(x['d'], self.pc)
        self.x = d

    def mcrals(self):
        options = {}
        options['mass'] = np.array([self.nonnegativeBox2.currentText(),
                                    self.unimodalityBox.currentText()])
        options['chrom'] = np.array([self.nonnegativeBox1.currentText(),
                                     self.unimodalityBox.currentText()])
        tolsigma = 0.1
        niter = 0
        idev = 0
        dn = self.x
        u, s, v, d, sd = pcarep(dn, self.pc)
        sstn = np.sum(np.power(dn, 2))
        sst = np.sum(np.power(d, 2))
        sigma2 = np.power(sstn, 0.5)
        nit = 30
        crs = self.pures
        while niter < nit:
            niter += 1
            conc, spec, res, resn, u, un, sigma, change, lof_pca, lof_exp, r2 = \
                mcr_als(d, dn, sst, sstn, sigma2, crs, options)

            self.axes2.clear()
            self.axes2.set_xlabel('scan')
            self.axes2.set_title('Resolved Chromatographic Profiles')
            self.axes2.plot(conc)
            self.redraw2()

            if change < 0.0:
                idev += 1
            else:
                idev = 0
            change = np.dot(100, change)
            lof_pca = np.power((u/sst), 0.5)*100
            lof_exp = np.power((un/sstn), 0.5)*100
            r2 = (sstn-un)/sstn
            if change > 0 or niter == 1:
                print(change)
                print(sigma)
                print(sigma2)
                sigma2 = sigma
                copt = conc
                sopt = spec
                itopt = niter+1
                sdopt = np.array([lof_pca, lof_exp])
                ropt = res
                r2opt = r2
            if abs(change) < tolsigma:
                print('CONVERGENCE IS ACHIEVED, STOP!!!')
                break
            if idev >= 20:
                print('FIT NOT IMPROVING FOR 20 TMES CONSECUTIVELY (DIVERGENCE?), STOP!!!')
                break
            crs = spec
            print(str(niter))
        self.r2opt = r2opt
        rts = self.rttic[np.sort(np.argmax(copt, axis=0))]
        index = np.argsort(np.argmax(copt, axis=0))
        for ind, val in enumerate(index):
            self.rt.append(rts[ind])
            ss = sopt[val, :]
            self.ms.append(ss/norm(ss))
        self.add_items(itopt, sdopt, r2opt)

    def add_items(self, itopt, sdopt, r2opt):
        self.optiterQtext.setText(str(itopt))
        self.lofpcaQtext.setText(str(sdopt[0]))
        self.lofexpQtext.setText(str(sdopt[1]))
        self.r2Qtext.setText(str(r2opt))

    def puremethod(self):
        del self.axes1.collections[:]
        self.VariableTable.clear()
        ftext = self.NLinedit.text()
        if self.DComboBox.currentText() == "Spectrum":
            xx = self.x.T
        else:
            xx = self.x
        pureV = pure(xx, self.pc, int(ftext))
        self.pures = pureV['SP']
        self.axes1.plot(self.pures.T)
        self.redraw1()
        for val, pu in enumerate(pureV['IMP']):
            puItem = QtGui.QTableWidgetItem(str(pu))
            puItem.setFlags(puItem.flags() ^ QtCore.Qt.ItemIsEditable)
            self.VariableTable.insertRow(val)
            self.VariableTable.setItem(val, 0, puItem)
        self.init = 1
        # self.buttonBoxOK.setEnabled(True)

    def get_resu(self):
        if len(self.rt):
            RESU = {"methods": "M", "ms": self.ms, 'rt': self.rt, 'mz': self.X['mz'], 'pc':self.pc,'R2': self.r2opt}
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
    window = MCRALSQDialg()
    # window.updata_data(m,3)
    window.show()
    sys.exit(app.exec_())