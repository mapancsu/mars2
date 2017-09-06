import sys
from PyQt4 import QtGui
import PyQt4.Qwt5 as Qwt
from PyQt4 import Qt
import sqlite3  
from PyQt4.QtCore import QVariant,QSize,QSettings,QPoint,SIGNAL,Qt,QFile,QString
from PyQt4.QtGui import (QMainWindow,QApplication,QFont,
                         QPen,QColor,QVBoxLayout,QWidget,QLabel,QLineEdit,
                         QHBoxLayout,QPushButton,QFileDialog,QToolTip,
                         QListWidget,QTreeWidget,QTreeWidgetItem,QDialog,QInputDialog)
import numpy as np
from scipy.io import netcdf
from scipy.sparse import coo_matrix
#import matplotlib.pyplot as plt

NIST_DBPath ='./Library/NIST2011.db'
MOL_DBPath ='./Library/NISTMol1.db'

class NistLibraryDialog(QtGui.QDialog):
    def __init__(self,num,parent=None):
        QtGui.QDialog.__init__(self, parent)
        settings = QSettings()
        size = settings.value("MainWindow/Size",QVariant(QSize(780,400))).toSize()
        self.setWindowTitle('NIST library Browser')
        self.row=num
        self.resize(size)
        self.initControls()
        self.readmol()
    def updatePlots(self):
        self.ID_label.setText("Entry No. %s Of 212964    Libaray Name:Nist2011 "%(self.id+1))
        self.ShowMolFile()
        self.showMs()
        self.readmol()
        
    
    def initControls(self):
        self.main_frame = QWidget()
        self.plot3 = Qwt.QwtPlot(self)
        self.plot3.setCanvasBackground(Qt.white)  
        self.plot3.enableAxis(Qwt.QwtPlot.yLeft, False)
        self.plot3.enableAxis(Qwt.QwtPlot.xBottom, False)
        self.plot1 = Qwt.QwtPlot(self)
        self.plot1.setCanvasBackground(Qt.white)

        top1_hbox = QHBoxLayout()
        SeacherButton  = QPushButton("&Select")
        FrontButton = QPushButton("<<")
        LaterButton = QPushButton(">>")
        self.connect(SeacherButton, SIGNAL('clicked()'), self.SeacherButton)
        self.connect(FrontButton, SIGNAL('clicked()'), self.FrontButton)
        self.connect(LaterButton, SIGNAL('clicked()'), self.LaterButton)
        FrontButton.setFixedWidth(50)
        SeacherButton.setFixedWidth(100)
        LaterButton.setFixedWidth(50)
        top1_hbox.addWidget(FrontButton)
        top1_hbox.addWidget(SeacherButton)
        top1_hbox.addWidget(LaterButton)
        self.id=self.row
        self.ID_label = QLabel("Entry No. %s Of 212964    Libaray Name:Nist2011 "%(self.id+1))
        top1_hbox.addWidget(self.ID_label)
        
        top2_hbox = QHBoxLayout()
        Name_label = QLabel("Name:")
        Name_label.setFixedWidth(70)
        self.Name_edit = QLineEdit()
        #Name_edit.setFixedWidth(150)
        top2_hbox.addWidget(Name_label)
        top2_hbox.addWidget(self.Name_edit)
               
        
        top3_hbox = QHBoxLayout()
        Formula_label = QLabel("Formula:")
        Formula_label.setFixedWidth(70)
        self.Formula_edit = QLineEdit()   
        #Name_edit.setFixedWidth(150)
        top3_hbox.addWidget(Formula_label)
        top3_hbox.addWidget(self.Formula_edit)    
        
        
        top4_hbox = QHBoxLayout()

        MW_label = QLabel("MW:")
        MW_label.setFixedWidth(70)
        self.MW_edit = QLineEdit()
        ExactMW_label = QLabel("Exact Mass:")
        ExactMW_label.setFixedWidth(70)
        self.ExactMW_edit = QLineEdit()    
        Cas_label = QLabel("CAS#:")
        Cas_label.setFixedWidth(70)
        self.Cas_edit = QLineEdit()
        Nist_label = QLabel("NIST#:")
        Nist_label.setFixedWidth(70)
        self.Nist_edit = QLineEdit()
        
        top4_hbox.addWidget(MW_label)
        top4_hbox.addWidget(self.MW_edit)
        top4_hbox.addWidget(ExactMW_label)
        top4_hbox.addWidget(self.ExactMW_edit)
        top4_hbox.addWidget(Cas_label)
        top4_hbox.addWidget(self.Cas_edit)
        top4_hbox.addWidget(Nist_label)
        top4_hbox.addWidget(self.Nist_edit) 
        top5_hbox = QHBoxLayout()
        Cont_label = QLabel("Contributor:")
        Cont_label.setFixedWidth(70)
        self.Cont_edit = QLineEdit()   
        #Name_edit.setFixedWidth(150)
        top5_hbox.addWidget(Cont_label)
        top5_hbox.addWidget(self.Cont_edit)
        top6_hbox = QHBoxLayout()
        Peak_label = QLabel("10 largest peaks:")
        Peak_label.setFixedWidth(100)
        self.Peak_edit = QLineEdit()   
        #Name_edit.setFixedWidth(150)
        top6_hbox.addWidget(Peak_label)
        top6_hbox.addWidget(self.Peak_edit) 
        top_Vbox = QVBoxLayout()
        top_Vbox.addLayout(top1_hbox)
        top_Vbox.addLayout(top2_hbox)
        top_Vbox.addLayout(top3_hbox)
        top_Vbox.addLayout(top4_hbox)
        top_Vbox.addLayout(top5_hbox)
        top_Vbox.addLayout(top6_hbox)
        
        below_hbox = QHBoxLayout()
        below_hbox.addWidget(self.plot1,3)
        below_hbox.addWidget(self.plot3,1)

        hbox = QVBoxLayout()
        hbox.addLayout(top_Vbox)
        hbox.addLayout(below_hbox)
        self.setLayout(hbox)
    def FrontButton(self):
        self.id=self.id-1
        self.updatePlots()
    def LaterButton(self):
        self.id=self.id+1
        self.updatePlots()
    def SeacherButton(self):
        newWidth, ok = QInputDialog.getInteger(self, self.tr("Select Entry"),
                                               self.tr("Entry range1-212964:"))
        if ok:
            self.id=newWidth
            self.updatePlots()
    def readmol(self):
        db = sqlite3.connect(NIST_DBPath)  
        self.cur = db.cursor()
        self.cur.execute("select name,Formula,MW,ExactMass,CAS,NIST,Comments,top10peakindex,top10peakintensity from catalog where id=%d"%(self.id))
        temp=self.cur.fetchall()
        self.Name_edit.setText(temp[0][0])
        self.Formula_edit.setText(temp[0][1])
        self.MW_edit.setText(str(temp[0][2]))
        self.ExactMW_edit.setText(temp[0][3])
        self.Cas_edit.setText(temp[0][4])
        self.Nist_edit.setText(temp[0][5])
        self.Cont_edit.setText(temp[0][6])
        self.top10peakindex = np.frombuffer(temp[0][7],dtype=np.int)
        self.top10peakintensity = np.frombuffer(temp[0][8],dtype=np.int)
        self.stringpeakindex='|'
        for j in range(len(self.top10peakindex)):
            self.stringpeakindex=self.stringpeakindex+str(self.top10peakindex[j])+" "+str(self.top10peakintensity[j])+"|"
        self.Peak_edit.setText(self.stringpeakindex)
        self.ShowMolFile()
        self.showMs()
    def ShowMolFile(self):
        self.plot3.clear()
        db = sqlite3.connect(MOL_DBPath)  
        cur = db.cursor()  
        Molecular = {}
        cur.execute("select * from catalog where id=%d"%self.id)
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
          
    def showMs(self):
        self.cur.execute("select name,peakindex,peakintensity,MW from catalog where id=%d"%self.id)
        temp=self.cur.fetchall()
        masstemp=np.frombuffer(temp[0][1],dtype=np.int)
        intensitytemp=np.frombuffer(temp[0][2],dtype=np.int)
        row = np.zeros(len(masstemp))
        mass = coo_matrix( (intensitytemp,(row,masstemp)), shape=(1,temp[0][3]+50)).todense()
        self.massSeacher=mass.tolist()
        radio_MS=0.01*(np.max(self.massSeacher[0])-np.min(self.massSeacher[0]))
        peak_MS=[]  
        for i in range(5,len(self.massSeacher[0])-5):
            if (self.massSeacher[0][i]==max(self.massSeacher[0][i-5:i+5]) and self.massSeacher[0][i]>=radio_MS):
                peak_intensity_MS=(i,self.massSeacher[0][i])
                peak_MS.append(peak_intensity_MS)
        self.plot1.clear()
#        self.plot1.setTitle("MS of %s"%str(temp[0][0][:-2]))
        self.plot1.setAxisScale(self.plot1.yLeft,0,1.1*np.max(self.massSeacher[0]))
        color = QColor('black')
        curve2 = Qwt.QwtPlotCurve("test1")
        pen = QPen(color)
        pen.setWidth(1)
        curve2.setPen(pen)
        self.axis= np.arange(temp[0][3]+50)+1
        curve2.setData(self.axis,self.massSeacher[0])
        curve2.setStyle(Qwt.QwtPlotCurve.Sticks)
        curve2.attach(self.plot1)
        for i in range(len(peak_MS)):
            text_MS=Qwt.QwtText('%s'%(str(peak_MS[i][0])))
            marker_MS = Qwt.QwtPlotMarker()
            marker_MS.setLabelAlignment(Qt.AlignCenter | Qt.AlignTop)
            marker_MS.setLabel(text_MS)
            marker_MS.setValue(peak_MS[i][0],peak_MS[i][1])
            marker_MS.attach(self.plot1)        
        self.plot1.replot()
        self.ShowMolFile()        
        
        