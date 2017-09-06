from PyQt4 import QtGui
from scipy import *
import PyQt4.Qwt5 as Qwt
from PyQt4.QtCore import Qt,QSettings,QVariant,QSize,SIGNAL,SLOT
from PyQt4.QtGui import (QVBoxLayout,QLabel,QTreeWidgetItem,QTreeWidget,QListWidget,QFont,
                         QHBoxLayout,QPen,QPushButton,QColor,QLineEdit,QTableWidget,QTableWidgetItem)
import numpy as np
import time
from math import floor 
import numpy.fft.fftpack as F
class ElemCompDialog(QtGui.QDialog):

    def __init__(self,mz, mass,xp,parent=None):
        QtGui.QDialog.__init__(self, parent)
        settings = QSettings()
        size = settings.value("MainWindow/Size",QVariant(QSize(1024,650))).toSize()
        self.resize(size)

        self.setWindowTitle('Elenmental Composition')
        self.xp=xp
        self.mass=mass
        self.mz=mz
        print mz
        self.initControls()
#        self.initPlots()
    def initControls(self):
        self.plot1 = Qwt.QwtPlot(self)
        self.plot1.setCanvasBackground(Qt.white)
        self.plot2 = Qwt.QwtPlot(self)
        self.plot2.setCanvasBackground(Qt.white)
        self.list = QTreeWidget()
        self.list.setColumnCount(11)
        self.list.setColumnWidth(0,80)
        self.list.setColumnWidth(1,80)
        self.list.setColumnWidth(2,60)
        self.list.setColumnWidth(3,60)
        self.list.setColumnWidth(4,60)
        self.list.setColumnWidth(5,150)
        self.list.setColumnWidth(7,30)
        self.list.setColumnWidth(8,30)
        self.list.setColumnWidth(9,30)
        self.list.setColumnWidth(10,30)
        self.list.setHeaderLabels(['Mass','Calc.Mass','mDa','PPM','DBE','Formula','Fit Conf %','C','H','N','O'])
        self.list.setSortingEnabled(True)
        
        self.table = QTableWidget(1,11)
        self.table.setColumnWidth(0,80)
        self.table.setColumnWidth(1,80)
        self.table.setColumnWidth(2,60)
        self.table.setColumnWidth(3,60)
        self.table.setColumnWidth(4,60)
        self.table.setColumnWidth(5,150)
        self.table.setColumnWidth(7,30)
        self.table.setColumnWidth(8,30)
        self.table.setColumnWidth(9,30)
        self.table.setColumnWidth(10,30)
        self.table.setHorizontalHeaderLabels(['Mass','Calc.Mass','mDa','PPM','DBE','Formula','Fit Conf %','C','H','N','O'])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  
        self.table.setSelectionBehavior(QTableWidget.SelectRows)  
        self.table.setSelectionMode(QTableWidget.SingleSelection)  
        self.table.setAlternatingRowColors(True)
        print self.connect(self.table, SIGNAL("itemActivated(QTableWidgetItem*)"), self.tableClicked)
#        self.connect(self.library_list, SIGNAL("itemSelectionChanged()"), self.libraryListClicked)
        up_hbox=QVBoxLayout()
        up_hbox.addWidget(self.table)
        down_hbox = QVBoxLayout()
        down_hbox.addWidget(self.plot1)
        down_hbox.addWidget(self.plot2)
        hbox = QVBoxLayout()
        hbox.addLayout(up_hbox, 3.5)
        hbox.addLayout(down_hbox, 3.5)
        self.setLayout(hbox)
        self.cal_mass()
    def tableClicked(self,item):
        self.C=self.table.item(item.row(),7).text()
        self.H=self.table.item(item.row(),8).text()
        self.N=self.table.item(item.row(),9).text()
        self.O=self.table.item(item.row(),10).text()
        self.cal_isotopic()
        self.initPlots()
    def initPlots(self):
        self.plot1.clear()
#        self.plot1.setTitle("Observed Isotope Distribution")
#        self.plot1.setAxisTitle(Qwt.QwtPlot.xBottom, 'Raman shift (cm-1)')
#        self.plot1.setAxisTitle(Qwt.QwtPlot.yLeft, 'Intensity')
#        grid = Qwt.QwtPlotGrid()
        pen = QPen(Qt.DotLine)
        pen.setColor(Qt.black)
        pen.setWidth(0)
#        grid.setPen(pen)
#        grid.attach(self.plot1)
        self.mass1=self.mass/self.massy*100
        self.plot1.setAxisScale(self.plot1.xBottom,self.x_min,self.x_max)
        self.plot1.setAxisScale(self.plot1.yLeft,0,1.1*100)
        color = QColor('black')
        curve = Qwt.QwtPlotCurve("test1")
        pen = QPen(color)
        pen.setWidth(1)
        curve.setPen(pen)
        curve.setData(self.mass_x,self.mass_y)
        curve.setStyle(Qwt.QwtPlotCurve.Sticks)
        curve.attach(self.plot1)  
        self.plot1.replot()
        
        self.plot2.clear()
#        self.plot2.setTitle("Theoretical Isotope Distribution")
#        self.plot2.setAxisTitle(Qwt.QwtPlot.xBottom, 'Raman shift (cm-1)')
#        self.plot2.setAxisTitle(Qwt.QwtPlot.yLeft, 'Intensity')
#        grid = Qwt.QwtPlotGrid()
        pen = QPen(Qt.DotLine)
        pen.setColor(Qt.blue)
        
        self.plot2.setAxisScale(self.plot1.xBottom,self.x_min,self.x_max)
        self.plot2.setAxisScale(self.plot1.yLeft,0,1.1*100)
        color = QColor('blue')
        curve = Qwt.QwtPlotCurve("test1")
        pen = QPen(color)
        pen.setWidth(1)
        curve.setPen(pen)
#        self.axis= np.arange(len(self.mass))
        curve.setData(self.x,self.y)
        curve.setStyle(Qwt.QwtPlotCurve.Sticks)
        curve.attach(self.plot2)  
        pen.setWidth(0)
#        grid.setPen(pen)
#        grid.attach(self.plot2)
        self.plot2.replot()

    def cal_mass(self):
#        charge=0.0
        tol = 10.0
#        electron=0.000549
        measured_mass=np.round(self.xp)
        limit_lo = measured_mass - (tol/1000.0)
        limit_hi = measured_mass + (tol/1000.0)
        mass=[]
        mass.append((12.000000000,2.0,"C"))
        #mass.append((78.9183376,-1.0,"Br"))
        #mass.append((34.96885271,-1.0,"Cl"))
        #mass.append((31.97207069,0.0,"S"))
        #mass.append((30.97376151,1.0,"P"))
        #mass.append((27.9769265327,2.0,"Si"))
        #mass.append((22.98976967,-1.0,"Na"))
        #mass.append((18.99840320,-1.0,"F"))
        mass.append((15.9949146221,0.0,"O"))
        mass.append((14.0030740052,1.0,"N"))
        mass.append((1.0078250321,-1.0,"H"))
        print range(1,10)
        print mass[0][0]
        print mass[1][0]
        print mass[2][0]
        calc_mass=[]
        for i in range(1,int(floor(measured_mass/mass[0][0]))+1):
            for j in range(0,int(floor((measured_mass-mass[0][0]*i)/mass[1][0]))+1):
                for k in range(0,int(floor((measured_mass-mass[0][0]*i-mass[1][0]*j)/mass[2][0]))+1):
        #            rr=(measured_mass-mass[0][0]*i-mass[1][0]*j-mass[2][0]*k)/mass[3][0]
        #            rrr=round((measured_mass-mass[0][0]*i-mass[1][0]*j-mass[2][0]*k)/mass[3][0])
        #            rrrr=int(round((measured_mass-mass[0][0]*i-mass[1][0]*j-mass[2][0]*k)/mass[3][0]))
        #            print "rr:%s"%rr+" rrr:%s"%rrr+" rrrr:%s"%rrrr
                    r=int(round((measured_mass-mass[0][0]*i-mass[1][0]*j-mass[2][0]*k)/mass[3][0]))
                    calmass=mass[0][0]*i+mass[1][0]*j+mass[2][0]*k+mass[3][0]*r
                    if (mass[0][1]*i+mass[2][1]*k+mass[3][1]*r)>=-1 and calmass>limit_lo and calmass<limit_hi:
                        calc_mass.append((calmass,i,j,k,r))
        print len(calc_mass)
        for ii in range(0,len(calc_mass)):
            mda=(measured_mass-calc_mass[ii][0])*1000
            ppm=(measured_mass-calc_mass[ii][0])/measured_mass*1000000
            DBE=(calc_mass[ii][1]*2+calc_mass[ii][3]-calc_mass[ii][4]+2)/2.0
            self.calmass="%.4f"%calc_mass[ii][0]
            self.mda="%.1f"%mda
            self.ppm="%.1f"%ppm
            self.DBE="%.1f"%DBE
            self.C="%s"%calc_mass[ii][1]
            self.H="%s"%calc_mass[ii][4]
            self.N="%s"%calc_mass[ii][3]
            self.O="%s"%calc_mass[ii][2]
            self.Formula="C%2s"%self.C+" H%2s"%self.H+" N%2s"%self.N+" O%2s"%self.O
            mass=str(self.xp)
#            if not(ii==0):
#                mass=""
            self.cal_isotopic()
            self.initPlots()
            self.conf="%.1f"%self.mass_diff
            self.table.insertRow(ii)
            self.table.setRowHeight(ii,20)
            self.table.setItem(ii, 0,QTableWidgetItem(mass))  
            self.table.setItem(ii, 1,QTableWidgetItem(self.calmass))
            self.table.setItem(ii, 2,QTableWidgetItem(self.mda))
            self.table.setItem(ii, 3,QTableWidgetItem(self.ppm))
            self.table.setItem(ii, 4,QTableWidgetItem(self.DBE))
            self.table.setItem(ii, 5,QTableWidgetItem(self.Formula))
            self.table.setItem(ii, 6,QTableWidgetItem(self.conf))
            self.table.setItem(ii, 7,QTableWidgetItem(self.C))
            self.table.setItem(ii, 8,QTableWidgetItem(self.H))
            self.table.setItem(ii, 9,QTableWidgetItem(self.N))
            self.table.setItem(ii, 10,QTableWidgetItem(self.O))

            item=QTreeWidgetItem([mass,str(self.calmass),str(self.mda),str(self.ppm),str(self.DBE),self.Formula,str(self.conf),str(self.C),str(self.H),str(self.N),str(self.O)])
            self.list.addTopLevelItem(item)
        self.table.removeRow(len(calc_mass))
        self.table.setSortingEnabled(True)
       
#        self.table.sortByColumn(1,Qt.DescendingOrder)
    def next2pow(self):
        return 2**int(np.ceil(np.log(float(self.xx))/np.log(2.0)))
    def cal_isotopic(self):
        MAX_ELEMENTS=5+1  # add 1 due to mass correction 'element'
        MAX_ISOTOPES=4    # maxiumum # of isotopes for one element
        CUTOFF=1e-4       # relative intensity cutoff for plotting
        
        WINDOW_SIZE = 500
        #WINDOW_SIZE=input('Window size (in Da) ---> ');
        
        #RESOLUTION=input('Resolution (in Da) ----> ');  % mass unit used in vectors
        RESOLUTION = 1
        if RESOLUTION < 0.00001:#  % minimal mass step allowed
          RESOLUTION = 0.00001
        elif RESOLUTION > 0.5:  # maximal mass step allowed
          RESOLUTION = 0.5
        
        R=0.00001/RESOLUTION#  % R is used to scale nuclide masses (see below)
        
        WINDOW_SIZE=WINDOW_SIZE/RESOLUTION; 
        self.xx=WINDOW_SIZE  # convert window size to new mass units
        WINDOW_SIZE=self.next2pow();  # fast radix-2 fast-Fourier transform algorithm
        
        if WINDOW_SIZE < np.round(496708*R)+1:
          WINDOW_SIZE = self.next2pow(np.round(496708*R)+1)  # just to make sure window is big enough
        
        
        #H378 C254 N65 O75 S6
        M=np.array([int(self.H),int(self.C),int(self.N),int(self.O),0,0]) #% empiric formula, e.g. bovine insulin
        
        # isotopic abundances stored in matrix A (one row for each element)
        A=np.zeros((MAX_ELEMENTS,MAX_ISOTOPES,2));
        
        A[0][0,:] = [100783,0.9998443]#                 % 1H
        A[0][1,:] = [201410,0.0001557]#                 % 2H
        A[1][0,:] = [100000,0.98889]#                   % 12C
        A[1][1,:] = [200336,0.01111]#                   % 13C
        A[2][0,:] = [100307,0.99634]#                   % 14N
        A[2][1,:] = [200011,0.00366]#                   % 15N
        A[3][0,:] = [99492,0.997628]#                  % 16O
        A[3][1,:] = [199913,0.000372]#                  % 17O
        A[3][2,:] = [299916,0.002000]#                  % 18O
        A[4][0,:] = [97207,0.95018]#                   % 32S
        A[4][1,:] = [197146,0.00750]#                   % 33S
        A[4][2,:] = [296787,0.04215]#                   % 34S
        A[4][3,:] = [496708,0.00017]#                   % 36S
        A[5][0,:] = [100000,1.00000]#                   % for shifting mass so that Mmi is
        #                                             % near left limit of window
        
        Mmi=np.array([np.round(100783*R), np.round(100000*R),\
                     np.round(100307*R),np.round(99492*R), np.round(97207*R), 0])*M#  % (Virtual) monoisotopic mass in new units
        Mmi = Mmi.sum()
        #% mass shift so Mmi is in left limit of window:
        FOLDED=np.floor(Mmi/(WINDOW_SIZE-1))+1#  % folded FOLDED times (always one folding due to shift below)
        #% shift distribution to 1 Da from lower window limit:
        M[MAX_ELEMENTS-1]=np.ceil(((WINDOW_SIZE-1)-np.mod(Mmi,WINDOW_SIZE-1)+np.round(100000*R))*RESOLUTION)
        
        MASS_REMOVED=np.array([0,11,13,15,31,-1])*M#% correction for 'virtual' elements and mass shift
        begin=WINDOW_SIZE*RESOLUTION+MASS_REMOVED.sum()
        end=2*(WINDOW_SIZE-1)*RESOLUTION+MASS_REMOVED.sum()
        
        ptA=np.ones(WINDOW_SIZE);
        t_fft=0
        t_mult=0
        
        for i in xrange(MAX_ELEMENTS):
        
            tA=np.zeros(WINDOW_SIZE)
            for j in xrange(MAX_ISOTOPES):
                if A[i][j,0] != 0:
                    #removed +1 after R)+1 --we're using python
                    tA[np.round(A[i][j,0]*R)]=A[i][j,1]#;  % put isotopic distribution in tA
            t0 = time.clock()
            tA=F.fft(tA) # FFT along elements isotopic distribution  O(nlogn)
            t_fft = time.clock()-t0
            t0 = time.clock()
            tA=tA**M[i]#  % O(n)
            #################
            ptA = ptA*tA#  % O(n)#this is where it is messing UP
            #################
            t1 = time.clock()
            t_mult=t1-t0
        

        t0=time.clock()
        ptA=F.ifft(ptA).real#;  % O(nlogn)

        t0=time.clock()
        
        MA=np.linspace(begin,end,WINDOW_SIZE-1)
        ind=np.where(ptA>CUTOFF)[0]
        
        self.x = MA[ind]
        self.y = ptA[ind]
        self.x_min=int(np.min(self.x)-(np.max(self.x)-np.min(self.x)))
        self.x_max=int(np.min(self.x)+(np.max(self.x)-np.min(self.x)))
        
        self.mass_y=np.ones(len(self.x))
        mass_diff=np.ones(len(self.x))
        
        mzInd= np.logical_and((self.mz>=self.x_min),(self.mz<=self.x_max))
        self.mass_y=self.mass[mzInd]
        self.mass_x=self.mz[mzInd]
        
        
#         for i in range(len(self.x)):
#             self.mass_y[i]=self.mass[int(self.x[i])]
        self.massy=np.max(self.mass_y)
        print self.massy
        self.mass_y=self.mass_y/max(self.mass_y)*100
        self.y=self.y/max(self.y)*100
#        k=(self.mass_y*self.y).sum()/(self.mass_y*self.mass_y).sum()
#        self.fit=((k*self.mass_y-self.y)*(k*self.mass_y-self.y)).sum()/(self.y*self.y).sum()
        for i in range(len(self.y)):
            mass_diff[i]=np.abs(self.mass_y[i]-self.y[i])/(self.mass_y[i]+self.y[i])
        self.mass_diff=(1-mass_diff.sum()/len(mass_diff))*100


    