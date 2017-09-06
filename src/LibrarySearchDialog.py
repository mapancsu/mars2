# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 19:00:04 2012

@author: Administrator
"""

from PyQt4 import QtGui
from scipy import *
import sqlite3
import PyQt4.Qwt5 as Qwt
from PyQt4.QtCore import Qt,QSettings,QVariant,QSize,SIGNAL,SLOT,QPoint
from PyQt4.QtGui import (QVBoxLayout,QLabel,QTreeWidgetItem,QTreeWidget,QListWidget,QFont,
                         QHBoxLayout,QPen,QPushButton,QColor,QLineEdit)
import numpy as np
from scipy.sparse import coo_matrix
import time
from ElementalComposition import ElemCompDialog
from NistLibrary import NistLibraryDialog
from sparseCorrelation import corrcoef_ms

from NIST import NISTSearch
from Mass import binMS

NIST_DBPath ='./Library/NIST2011.db'
MOL_DBPath ='./Library/NISTMol1.db'

def processMS(ms):
    mz=ms['mz']
    val=ms['val']
    ms_scaled=100*val/np.max(val)
    
    inds=np.argsort(val)[::-1]
    mz_top10  = mz[inds]
    val_top10 = ms_scaled[inds]
    
    peak_MS=[]
    for i in range(min(len(inds),10)):
            peak_intensity_MS=(mz_top10[i],val_top10[i])
            peak_MS.append(peak_intensity_MS)
    return (mz,ms_scaled,peak_MS)

def binMS_ZDJ(ms):
    mz=ms['mz']
    val=ms['val']
    rg=np.linspace(floor(mz[0])-0.5, ceil(mz[-1])+0.5,num=ceil(mz[-1])-floor(mz[0])+2)
    r=int(ceil(mz[-1])-floor(mz[0])+1)
    mz_bin  = np.zeros((r,),dtype=int)
    val_bin = np.zeros((r,),dtype=int)
    inds=np.searchsorted(mz, rg)
    for i in range(0,r):
        mz_bin[i] = (rg[i]+rg[i+1])/2
        val_bin[i]= sum(val[inds[i]:inds[i+1]])
    return {'mz':mz_bin, 'val':val_bin}

class LibrarySearchDialog(QtGui.QDialog):

    def __init__(self,ms,parent=None):
        QtGui.QDialog.__init__(self, parent)
        settings = QSettings()
        size = settings.value("MainWindow/Size",QVariant(QSize(1024,600))).toSize()
        self.resize(size)

        self.setWindowTitle('Identify unknown compound by NIST 2011 Library')
        self.ms=ms
        pms=processMS(ms)
        self.axis=pms[0]
        self.mass=pms[1]
        self.peak_MS=pms[2]
        self.maxmass=int(self.axis[-1])
        
        self.initControls()
        self.initPlots()

        self.selectedIndex=[]
        self.percent=[]
        self._cpointsPicker1 = Qwt.QwtPicker(self.plot1.canvas())
        self._cpointsPicker1.setSelectionFlags(Qwt.QwtPicker.PointSelection)
        self._cpointsPicker1.widgetMouseDoubleClickEvent = self.plot1MouseDoubleClickEvent
    def updatePlots(self):
        
        reply = QtGui.QMessageBox.question(self, 'EleComp Parameters',
            "Are you sure to Use Single Mass %s?"%self.selected_mz, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
#            print self.x
            self.ElemCompDialog1()
#        self.m.setLabel(self.text1)
#       self.m.setValue(self.x, 0.0)
#        rowMs=int((self.x-self.scan_acquisition_time.data[0])/0.2)
#        self.showMs(rowMs)
        
#        self.plot1.replot()
    def plot1MouseDoubleClickEvent(self, event):
        self.xp = self.plot1.invTransform(Qwt.QwtPlot.xBottom, event.x())
        n1=np.searchsorted(self.axis, self.xp-1)
        n2=np.searchsorted(self.axis, self.xp+1)
        self.selected_mz=self.axis[np.argmax(self.mass[n1:n2])+n1]
        print self.xp
        print self.selected_mz
        self.updatePlots()
    def ElemCompDialog1(self):
        dialog = ElemCompDialog(mz=self.axis,mass=self.mass,xp=self.selected_mz)
        dialog.move(QPoint(100, 10))
        res=dialog.exec_()
    def initControls(self):
        self.plot3 = Qwt.QwtPlot(self)
        self.plot3.setCanvasBackground(Qt.white)  
        self.plot3.enableAxis(Qwt.QwtPlot.yLeft, False)
        self.plot3.enableAxis(Qwt.QwtPlot.xBottom, False)
        self.plot1 = Qwt.QwtPlot(self)
        self.plot1.setCanvasBackground(Qt.white)
        self.plot2 = Qwt.QwtPlot(self)
        self.plot2.setCanvasBackground(Qt.white)
        
        library_label = QLabel("MS in NIST Library:")
        self.library_list = QTreeWidget()
        self.library_list.setColumnCount(3)
        self.library_list.setHeaderLabels(['No.','Similarity','Mol Wt','Formula','Name'])
        self.library_list.setSortingEnabled(False)
        self.connect(self.library_list, SIGNAL("itemSelectionChanged()"), self.libraryListClicked)
        self.connect(self.library_list, SIGNAL("itemActivated (QTreeWidgetItem *,int)"), self.libraryListDoubleClicked)
        mxiture_label = QLabel("Molecular structure :")
#        self.mixture_list = QListWidget()
        
        okButton  = QPushButton("&Search")
        self.connect(okButton,SIGNAL("clicked()"),self.Seach)

        
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.plot1)
        left_vbox.addWidget(self.plot2)
        
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(library_label)
        right_vbox.addWidget(self.library_list)
        
        hboxPercent = QHBoxLayout()
#        hboxPercent.addWidget(percent_label)
#        hboxPercent.addWidget(self.edit_percent)
        right_vbox.addLayout(hboxPercent)
        
#        right_vbox.addWidget(self.add_button)
        right_vbox.addWidget(mxiture_label)
        right_vbox.addWidget(self.plot3)
        right_vbox.addWidget(okButton)
        
        hbox = QHBoxLayout()
        hbox.addLayout(left_vbox, 2.5)
        hbox.addLayout(right_vbox, 1.5)
        self.setLayout(hbox)

        #self.setCentralWidget(self.main_frame) 

    def initPlots(self):
        self.plot3.clear()
        self.plot3.setAxisScale(self.plot1.xBottom,-4,4)
        self.plot3.setAxisScale(self.plot1.yLeft,-4,4)
        self.plot1.clear()
        self.plot1.setTitle("Search MS")
#        self.plot1.setAxisTitle(Qwt.QwtPlot.xBottom, 'Raman shift (cm-1)')
        self.plot1.setAxisTitle(Qwt.QwtPlot.yLeft, 'Intensity')
        grid = Qwt.QwtPlotGrid()
        pen = QPen(Qt.DotLine)
        pen.setColor(Qt.black)
        pen.setWidth(0)
        grid.setPen(pen)
        grid.attach(self.plot1)
        self.plot1.setAxisScale(self.plot1.yLeft,0,1.1*np.max(self.mass))
        color = QColor('black')
        curve = Qwt.QwtPlotCurve("test1")
        pen = QPen(color)
        pen.setWidth(1)
        curve.setPen(pen)
        #self.axis= np.arange(len(self.mass))
        curve.setData(self.axis,self.mass)
        curve.setStyle(Qwt.QwtPlotCurve.Sticks)
        curve.attach(self.plot1)
        for i in range(len(self.peak_MS)):
            text_MS=Qwt.QwtText('%s'%(str(self.peak_MS[i][0])))
            marker_MS = Qwt.QwtPlotMarker()
            marker_MS.setLabelAlignment(Qt.AlignCenter | Qt.AlignTop)
            marker_MS.setLabel(text_MS)
            marker_MS.setValue(self.peak_MS[i][0],self.peak_MS[i][1])
            marker_MS.attach(self.plot1)    
        self.plot1.replot()
        
        self.plot2.clear()
        self.plot2.setTitle("NIST MS")
#        self.plot2.setAxisTitle(Qwt.QwtPlot.xBottom, 'Raman shift (cm-1)')
        self.plot2.setAxisTitle(Qwt.QwtPlot.yLeft, 'Intensity')
        grid = Qwt.QwtPlotGrid()
        pen = QPen(Qt.DotLine)
        pen.setColor(Qt.black)
        pen.setWidth(0)
        grid.setPen(pen)
        grid.attach(self.plot2)
        self.plot2.replot()

    def libraryListClicked(self):
        
        row = self.library_list.indexOfTopLevelItem(self.library_list.currentItem())
        self.row1=self.masscor[row][0]
        print 'row: %s'%self.row1
        self.showMs(self.row1)
    def libraryListDoubleClicked(self, item ,pos):
        dialog = NistLibraryDialog(num=self.row1)
        dialog.move(QPoint(100, 10))
        res=dialog.exec_()
    def showMs(self,row1):
        self.row1=row1
        self.cur.execute("select name,peakindex,peakintensity  from catalog where id=%d"%row1)
        temp=self.cur.fetchall()
        masstemp=np.frombuffer(temp[0][1],dtype=np.int)
        intensitytemp=np.frombuffer(temp[0][2],dtype=np.int)
        intensitytemp=100*intensitytemp/np.max(intensitytemp)
        row = np.zeros(len(masstemp))
        mass = coo_matrix( (intensitytemp,(row,masstemp)), shape=(1,ceil(masstemp[-1])+1)).todense()
        self.massSeacher=mass.tolist()
        radio_MS=0.01*(np.max(self.massSeacher[0])-np.min(self.massSeacher[0]))
        peak_MS=[]  
        for i in range(5,len(self.massSeacher[0])-5):
            if (self.massSeacher[0][i]==max(self.massSeacher[0][i-5:i+5]) and self.massSeacher[0][i]>=radio_MS):
                peak_intensity_MS=(i,self.massSeacher[0][i])
                peak_MS.append(peak_intensity_MS)
        self.plot2.clear()
        self.plot2.setTitle("MS of %s"%str(temp[0][0][:-2]))
        color = QColor('black')
        curve2 = Qwt.QwtPlotCurve("test1")
        pen = QPen(color)
        pen.setWidth(1)
        curve2.setPen(pen)
        self.axis2= masstemp
        curve2.setData(masstemp,intensitytemp)
        curve2.setStyle(Qwt.QwtPlotCurve.Sticks)
        curve2.attach(self.plot2)
        for i in range(len(peak_MS)):
            text_MS=Qwt.QwtText('%s'%(str(peak_MS[i][0])))
            marker_MS = Qwt.QwtPlotMarker()
            marker_MS.setLabelAlignment(Qt.AlignCenter | Qt.AlignTop)
            marker_MS.setLabel(text_MS)
            marker_MS.setValue(peak_MS[i][0],peak_MS[i][1])
            marker_MS.attach(self.plot2)
        
        
        x=np.hstack((self.axis,self.axis2))
        x_min=np.min(x)
        x_max=np.max(x)
        y1_max=np.max(self.mass)
        y1_min=np.min(self.mass)
        y2_max=np.max(intensitytemp)
        y2_min=np.min(intensitytemp)
        
        self.plot1.setAxisScale(self.plot1.xBottom,0,x_max*1.1)
        self.plot2.setAxisScale(self.plot1.xBottom,0,x_max*1.1)
        self.plot1.setAxisScale(self.plot1.yLeft,0,y1_max*1.1)
        self.plot2.setAxisScale(self.plot1.yLeft,0,y2_max*1.1)
        
        self.plot1.replot()
        self.plot2.replot()
        self.ShowMolFile()

    def comnum(self):
        i=0
        for x in self.findms:
            if x in self.basems:
                i=i+1
        cor=i*2.0/(len(self.findms)+len(self.basems))
        return cor


    def Seach_ZDJ(self):
        self.findms=[]
        (mz_bin,val_bin,peaks_bin)=processMS(binMS(self.ms))
        peaks_bin.sort(key=lambda d:d[1],reverse=True)
        self.peak_index=[]    
        self.peak_inten=[]
        for i in range(0,len(peaks_bin)):
            self.peak_index.append(int(peaks_bin[i][0]))
            self.peak_inten.append(peaks_bin[i][1]) 
        if len(peaks_bin)<10:
            self.findms=self.peak_index
        else:
            self.findms=self.peak_index[0:10]
        time0=time.time()
        db = sqlite3.connect(NIST_DBPath)  
        self.cur = db.cursor()  
        self.cur.execute("select id,top10peakindex from catalog where MW>%d-28 and MW<%d+28"%(self.maxmass,self.maxmass))
        self.c=self.cur.fetchall()
        ms=[]
  
        for i in range(len(self.c)):
            self.basems=np.frombuffer(self.c[i][1],dtype=np.int)
            cor=self.comnum()
            if cor>0.4:
                temp=(self.c[i][0],cor)
                ms.append(temp)
        print ms
        self.masscor=[]
        tic=time.time()
        for i in range(len(ms)):
            self.cur.execute("select peakindex, peakintensity  from catalog where id=%d"%(ms[i][0]))
            temp=self.cur.fetchall()
            masstemp=np.frombuffer(temp[0][0],dtype=np.int)
            intensitytemp=np.frombuffer(temp[0][1],dtype=np.int)
            temp2=(ms[i][0],corrcoef_ms(masstemp, intensitytemp, mz_bin, val_bin))
            self.masscor.append(temp2)
        print time.time()-time0
        print time.time()-tic
        self.masscor.sort(key=lambda d:d[1],reverse=True)
        for i in range(min(25,len(ms))):
            self.cur.execute("select name,Formula,MW  from catalog where id=%d"%self.masscor[i][0])
            temp=self.cur.fetchall()
            strsimilarity='%.2f'%self.masscor[i][1]
            temp3=temp[0][1][:-1]
            temp1= temp[0][2]
            temp2=temp[0][0][:-2]
            item=QTreeWidgetItem([str(i+1),str(strsimilarity),str(temp1),str(temp3),str(temp2)])
            self.library_list.addTopLevelItem(item)   

        if len(ms)>0:
            self.showMs(self.masscor[0][0])
        
        zoomer=Qwt.QwtPlotZoomer(Qwt.QwtPlot.xBottom,
                                    Qwt.QwtPlot.yLeft,
                                    Qwt.QwtPicker.DragSelection,
                                    Qwt.QwtPicker.AlwaysOn,
                                    self.plot1.canvas())
        zoomer.setRubberBandPen(QPen(Qt.black))
        zoomer.setTrackerPen(QPen(Qt.blue))       
        self.plot1.zoomer = zoomer        
        self.plot1.zoomer.setZoomBase()

    def Seach(self):
      
        ms=self.ms
        nist=NISTSearch(NIST_DBPath)
        nist.top10_screen(self.ms)
        nist.corr()
        self.masscor=nist.corrs


        db = sqlite3.connect(NIST_DBPath)  
        self.cur = db.cursor()
        for i in range(min(25,len(self.masscor))):
            self.cur.execute("select name,Formula,MW  from catalog where id=%d"%self.masscor[i][0])
            temp=self.cur.fetchall()
            strsimilarity='%.2f'%self.masscor[i][1]
            temp3=temp[0][1][:-1]
            temp1= temp[0][2]
            temp2=temp[0][0][:-2]
            item=QTreeWidgetItem([str(i+1),str(strsimilarity),str(temp1),str(temp3),str(temp2)])
            self.library_list.addTopLevelItem(item)   

        if len(ms)>0:
            self.showMs(self.masscor[0][0])
        
        zoomer=Qwt.QwtPlotZoomer(Qwt.QwtPlot.xBottom,
                                    Qwt.QwtPlot.yLeft,
                                    Qwt.QwtPicker.DragSelection,
                                    Qwt.QwtPicker.AlwaysOn,
                                    self.plot1.canvas())
        zoomer.setRubberBandPen(QPen(Qt.black))
        zoomer.setTrackerPen(QPen(Qt.blue))       
        self.plot1.zoomer = zoomer        
        self.plot1.zoomer.setZoomBase()


    def ShowMolFile(self):
        self.plot3.clear()
        self.ID=self.row1
        Molecular = {}
        db = sqlite3.connect(MOL_DBPath)  
        cur = db.cursor()  
        cur.execute("select * from catalog where id=%d"%self.ID)
        c=cur.fetchall()
        Molecular["MolName"]=c[0][1]
        Molecular["MolNum"]=c[0][2]
        Molecular["MolBondNum"]=c[0][3]
        Molecular["Mol"]=c[0][4].split()
        Molecular["MolXAxis"]=np.frombuffer(c[0][5],dtype=np.float)
        Molecular["MolYAxis"]=np.frombuffer(c[0][6],dtype=np.float)
        Molecular["MolStyle"]=np.frombuffer(c[0][7],dtype=np.int)
        Molecular["bondX"]=np.frombuffer(c[0][8],dtype=np.int)
        Molecular["bondY"]=np.frombuffer(c[0][9],dtype=np.int)
        Molecular["bondNum"]=np.frombuffer(c[0][10],dtype=np.int)
        self.Molecular=Molecular
        color = QColor('black')
        curve = Qwt.QwtPlotCurve()
        pen = QPen(color)
        pen.setWidth(1)
        curve.setPen(pen)
        curve.setStyle(Qwt.QwtPlotCurve.NoCurve)
        curve.setSymbol(Qwt.QwtSymbol(Qwt.QwtSymbol.Ellipse, Qt.black,
              QPen(Qt.black), QSize(3,3)))
        curve.attach(self.plot3)
        curve.setData(self.Molecular["MolXAxis"],self.Molecular["MolYAxis"])
        tempstyl1=[]
        tempstyl2=[]
        tempstyl3=[]
        tempstyl4=[]
        for i in range(Molecular["MolBondNum"]):
            if Molecular["bondNum"][i]==1 and Molecular["MolStyle"][Molecular["bondX"][i]-1]==0 and Molecular["MolStyle"][Molecular["bondY"][i]-1]==0:
                tempstyl2.append(Molecular["bondX"][i])
                tempstyl2.append(Molecular["bondY"][i])
        for i in range(Molecular["MolBondNum"]):                    
            if Molecular["bondNum"][i]==2 and Molecular["MolStyle"][Molecular["bondX"][i]-1]==0 and Molecular["MolStyle"][Molecular["bondY"][i]-1]==0:
                if (Molecular["bondX"][i] in tempstyl2) and (Molecular["bondY"][i] in tempstyl2):
                    tempstyl1.append(Molecular["bondX"][i])
                    tempstyl1.append(Molecular["bondY"][i])
        for i in range(len(tempstyl2)/2):
            if (tempstyl2[2*i] in tempstyl1) and (tempstyl2[2*i+1] in tempstyl1):
                    tempstyl3.append(tempstyl2[2*i])
                    tempstyl3.append(tempstyl2[2*i+1])
        for i in range(len(tempstyl1)/2):
            if (tempstyl1[2*i] in tempstyl3) and (tempstyl1[2*i+1] in tempstyl3):
                    tempstyl4.append(tempstyl1[2*i])
                    tempstyl4.append(tempstyl1[2*i+1])
        tempstyl6=[]
        for i in range(len(tempstyl3)/2):
            if (tempstyl3[2*i] in tempstyl4) and (tempstyl3[2*i+1] in tempstyl4):
                    tempstyl6.append(tempstyl3[2*i])
                    tempstyl6.append(tempstyl3[2*i+1])            
        tempstyl5=[]
#            print tempstyl4
        while True:
            if len(tempstyl6)==0 or len(tempstyl4)==0:
                    break
            for i in range(len(tempstyl4)/2):
#                print i
                if not(tempstyl4[2*i] in tempstyl5):
                    tempindex3=tempstyl6.index(tempstyl4[2*i])
                    tempindex4=tempstyl6.index(tempstyl4[2*i+1])
                    temp1=tempstyl4[2*i]
                    temp2=tempstyl4[2*i+1]
                    if tempindex3%2==0:
                        temp3=tempstyl6[tempindex3+1]
                        tempindex3other=tempindex3+1
                    else:
                        temp3=tempstyl6[tempindex3-1]
                        tempindex3other=tempindex3-1
                    if tempindex4%2==0:
                        temp4=tempstyl6[tempindex4+1]
                        tempindex4other=tempindex4+1
                    else:
                        temp4=tempstyl6[tempindex4-1]
                        tempindex4other=tempindex4-1
                    tempindex5=tempstyl4.index(temp3)
                    tempindex6=tempstyl4.index(temp4)
                    if tempindex5%2==0:
                        temp5=tempstyl4[tempindex5+1]
                    else:
                        temp5=tempstyl4[tempindex5-1]
                    if tempindex6%2==0:
                        temp6=tempstyl4[tempindex6+1]
                    else:
                        temp6=tempstyl4[tempindex6-1]
                    tempindex7=tempstyl6.index(temp5)
                    if tempindex7%2==0:
                        temp7=tempstyl6[tempindex7+1]
                        tempindex7other=tempindex7+1
                    else:
                        temp7=tempstyl6[tempindex7-1]
                        tempindex7other=tempindex7-1
                    if temp7==temp6:
                        if not((temp1 in tempstyl5)and(temp2 in tempstyl5)and(temp3 in tempstyl5)and(temp4 in tempstyl5)and(temp5 in tempstyl5)and(temp6 in tempstyl5)):
                            tempstyl5.append(temp1)
                            tempstyl5.append(temp2)
                            tempstyl5.append(temp4)
                            tempstyl5.append(temp3)
                            tempstyl5.append(temp6)
                            tempstyl5.append(temp5)
                            temp=[tempindex3,tempindex3other,tempindex4,tempindex4other,tempindex7,tempindex7other]
                            temp.sort(reverse=True)
                            del tempstyl6[temp[0]]
                            del tempstyl6[temp[1]]
                            del tempstyl6[temp[2]]
                            del tempstyl6[temp[3]]
                            del tempstyl6[temp[4]]
                            del tempstyl6[temp[5]]
                            for i in np.arange((len(tempstyl4)-1)/2,-1,-1):
                                if not(tempstyl4[2*i] in tempstyl6) or not(tempstyl4[2*i+1] in tempstyl6):
                                        del tempstyl4[2*i+1]
                                        del tempstyl4[2*i]                                        
                            for i in np.arange((len(tempstyl6)-1)/2,-1,-1):
                                if not(tempstyl6[2*i] in tempstyl4) or not(tempstyl6[2*i+1] in tempstyl4):
                                        del tempstyl6[2*i+1]
                                        del tempstyl6[2*i]
                            for i in np.arange((len(tempstyl4)-1)/2,-1,-1):
                                if not(tempstyl4[2*i] in tempstyl6) or not(tempstyl4[2*i+1] in tempstyl6):
                                        del tempstyl4[2*i+1]
                                        del tempstyl4[2*i]
                            for i in np.arange((len(tempstyl6)-1)/2,-1,-1):
                                if not(tempstyl6[2*i] in tempstyl4) or not(tempstyl6[2*i+1] in tempstyl4):
                                        del tempstyl6[2*i+1]
                                        del tempstyl6[2*i]
                            break
#            tempstylCom=list(set(tempstyl1) & set(tempstyl2))            
#            styl=np.setdiff1d(tempstyl1,tempstylCom)
        for i in range(Molecular["MolBondNum"]):
            x1=self.Molecular["MolXAxis"][self.Molecular["bondX"][i]-1]
            x2=self.Molecular["MolXAxis"][self.Molecular["bondY"][i]-1]
            y1=self.Molecular["MolYAxis"][self.Molecular["bondX"][i]-1]
            y2=self.Molecular["MolYAxis"][self.Molecular["bondY"][i]-1]
            if (y2-y1)==0:
                Xdiff=0
                Ydiff=np.sqrt(0.003)
            else:
                h=(x2-x1)/(y2-y1)
                Xdiff=np.sqrt(0.003/(h*h+1))
                Ydiff=Xdiff*h                     
            if (Molecular["bondNum"][i]==2) and not(Molecular["bondX"][i] in tempstyl5):
                    tempx1=[]
                    tempy1=[]
                    tempx2=[]
                    tempy2=[]
                    tempx1.append(x1+Xdiff)
                    tempx1.append(x2+Xdiff)
                    tempy1.append(y1-Ydiff)
                    tempy1.append(y2-Ydiff)
                    tempx2.append(x1-Xdiff)
                    tempx2.append(x2-Xdiff)
                    tempy2.append(y1+Ydiff)
                    tempy2.append(y2+Ydiff)                    
                    curve2 = Qwt.QwtPlotCurve()
                    curve2.setStyle(Qwt.QwtPlotCurve.Lines)
                    curve2.attach(self.plot3)
                    curve2.setData(tempx1,tempy1)
                    curve3 = Qwt.QwtPlotCurve()
                    curve3.setStyle(Qwt.QwtPlotCurve.Lines)
                    curve3.attach(self.plot3)
                    curve3.setData(tempx2,tempy2)
            elif (Molecular["bondNum"][i]==3):
                    tempx=[]
                    tempy=[]
                    tempx.append(x1)
                    tempx.append(x2)
                    tempy.append(y1)
                    tempy.append(y2)
                    curve1 = Qwt.QwtPlotCurve()
                    curve1.setStyle(Qwt.QwtPlotCurve.Lines)
                    curve1.attach(self.plot3)                 
                    curve1.setData(tempx,tempy)
                    tempx1=[]
                    tempy1=[]
                    tempx2=[]
                    tempy2=[]
                    tempx1.append(x1+Xdiff)
                    tempx1.append(x2+Xdiff)
                    tempy1.append(y1-Ydiff)
                    tempy1.append(y2-Ydiff)
                    tempx2.append(x1-Xdiff)
                    tempx2.append(x2-Xdiff)
                    tempy2.append(y1+Ydiff)
                    tempy2.append(y2+Ydiff)                    
                    curve2 = Qwt.QwtPlotCurve()
                    curve2.setStyle(Qwt.QwtPlotCurve.Lines)
                    curve2.attach(self.plot3)
                    curve2.setData(tempx1,tempy1)
                    curve3 = Qwt.QwtPlotCurve()
                    curve3.setStyle(Qwt.QwtPlotCurve.Lines)
                    curve3.attach(self.plot3)
                    curve3.setData(tempx2,tempy2) 
            else :
                    tempx=[]
                    tempy=[]
                    tempx.append(x1)
                    tempx.append(x2)
                    tempy.append(y1)
                    tempy.append(y2)
                    curve1 = Qwt.QwtPlotCurve()
                    curve1.setStyle(Qwt.QwtPlotCurve.Lines)
                    curve1.attach(self.plot3)                 
                    curve1.setData(tempx,tempy)
        t=np.linspace(0,np.pi*2,100)
        diffx1=np.sin(t)*0.3
        diffy1=np.cos(t)*0.3
        for i in range(len(tempstyl5)/6):
            x0=0
            y0=0
            diffx=[]
            diffy=[]
            x0=Molecular["MolXAxis"][tempstyl5[6*i]-1]+Molecular["MolXAxis"][tempstyl5[6*i+1]-1]
            x0=x0+Molecular["MolXAxis"][tempstyl5[6*i+2]-1]+Molecular["MolXAxis"][tempstyl5[6*i+3]-1]
            x0=x0+Molecular["MolXAxis"][tempstyl5[6*i+4]-1]+Molecular["MolXAxis"][tempstyl5[6*i+5]-1]
            x0=x0/6
            y0=Molecular["MolYAxis"][tempstyl5[6*i]-1]+Molecular["MolYAxis"][tempstyl5[6*i+1]-1]
            y0=y0+Molecular["MolYAxis"][tempstyl5[6*i+2]-1]+Molecular["MolYAxis"][tempstyl5[6*i+3]-1]
            y0=y0+Molecular["MolYAxis"][tempstyl5[6*i+4]-1]+Molecular["MolYAxis"][tempstyl5[6*i+5]-1]
            y0=y0/6
            for i in range(len(diffx1)):
                diffx.append(diffx1[i]+x0)
                diffy.append(diffy1[i]+y0)
            curve4 = Qwt.QwtPlotCurve()
            curve4.setStyle(Qwt.QwtPlotCurve.Lines)
            curve4.attach(self.plot3)                 
            curve4.setData(diffx,diffy)            
        for i in range(Molecular["MolNum"]):
            if Molecular["MolStyle"][i]!=0:
                text=Qwt.QwtText('%s'%Molecular["Mol"][i])
#                    text=Qwt.QwtText('%s'%str(i+1))
                text.setColor(Qt.blue)
                text.setFont(QFont("Sans", 12))
                text.setBackgroundBrush(Qt.white)
                marker = Qwt.QwtPlotMarker()
                marker.setLabelAlignment(Qt.AlignCenter | Qt.AlignCenter)
                marker.setLabel(text)
                marker.setValue(self.Molecular["MolXAxis"][i],self.Molecular["MolYAxis"][i])
                marker.attach(self.plot3)
        self.plot3.setAxisScale(self.plot3.xBottom,min((min(Molecular["MolXAxis"])-0.5),-4),max((max(Molecular["MolXAxis"])+0.5),4))
        self.plot3.setAxisScale(self.plot3.yLeft,min((min(Molecular["MolYAxis"])-0.5),-4),max((max(Molecular["MolYAxis"])+0.5),4))
        self.plot3.replot()         
        
class NistLibraryDialog1(QtGui.QDialog):

    def __init__(self,parent=None):
        QtGui.QDialog.__init__(self, parent)
        settings = QSettings()
        size = settings.value("MainWindow/Size",QVariant(QSize(512,300))).toSize()
        self.resize(size)

        self.setWindowTitle('NIST library')
        self.initControls()
    def initControls(self):
        self.plot3 = Qwt.QwtPlot(self)
        self.plot3.setCanvasBackground(Qt.white)  
        self.plot3.enableAxis(Qwt.QwtPlot.yLeft, False)
        self.plot3.enableAxis(Qwt.QwtPlot.xBottom, False)
        self.plot1 = Qwt.QwtPlot(self)
        self.plot1.setCanvasBackground(Qt.white)
        SeacherButton  = QPushButton("&Seacher")
        FrontButton = QPushButton("<<")
        LaterButton = QPushButton(">>")
        top_hbox = QHBoxLayout()
        top_hbox.addWidget(FrontButton)
        top_hbox.addWidget(SeacherButton)
        top_hbox.addWidget(LaterButton)
        below_hbox = QHBoxLayout()
        below_hbox.addWidget(self.plot1)
        below_hbox.addWidget(self.plot3)

        hbox = QVBoxLayout()
        hbox.addLayout(top_hbox, 2.5)
        hbox.addLayout(below_hbox, 1.5)
        self.setLayout(hbox)
        
        
        