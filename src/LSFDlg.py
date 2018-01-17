__author__ = 'Administrator'


import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
import numpy as np

class LSFQDialg(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()
        self.segments = []

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

        self.resize(800, 600)
        self.move(320, 75)
        self.setWindowTitle("Least Square Fitting (2D)")

    def create_canvas(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = plt.subplot(111)
        self.axes.set_xlabel("Scans")
        self.axes.set_ylabel("Instensity")
        self.axes.set_title("Least Square Fitting(2D)", fontsize=9)
        self.axes.tick_params(axis='both', labelsize=8)
        plt.subplots_adjust(bottom=0.22, top=0.90, left=0.08, right=0.9)

        self.span = SpanSelector(self.axes, self.span_select_callback, 'horizontal', minspan=0.002, useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'),
                                 onmove_callback=None,
                                 button=[1])

        axspan = plt.axes([0.09, 0.04, 0.08, 0.075])
        axundo = plt.axes([0.2, 0.04, 0.08, 0.075])
        axstar = plt.axes([0.31, 0.04, 0.08, 0.075])
        self.btnspan = Button(axspan, 'span')
        self.btnundo = Button(axundo, 'undo')
        self.btnstar = Button(axstar, 'start')

        self.btnspan.on_clicked(self.span_mode)
        self.btnundo.on_clicked(self.undo_mode)
        self.btnstar.on_clicked(self.star_mode)

        self.span.set_active(True)
        self.redraw()
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]

    def redraw(self):
        self.canvas.draw()
        self.update()

    def span_mode(self, event):
        self.span.connect_event('motion_notify_event', self.span.onmove)
        self.span.connect_event('button_press_event', self.span.press)
        self.span.connect_event('button_release_event', self.span.release)
        self.span.connect_event('draw_event', self.span.update_background)
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw()
        print "span"

    def undo_mode(self, event):
        self.span.disconnect_events()
        if len(self.segments) >= 1:
            del self.axes.collections[:]
            self.segments = self.segments[:-1]
            self.select_inter(self.segments)
            self.redraw()
        print "undo"

    def star_mode(self, event):
        if len(self.segments) == 0:
            msgBox = QMessageBox()
            msgBox.setText("No selected noise region")
            msgBox.exec_()
        else:
            fit, bas = self.backremv(self.segments)
            self.show_bas(bas)
            self.show_fit(fit)
            self.emit(SIGNAL('after_baseline'), fit)

    def show_bas(self, bas):
        self.axes.plot(np.sum(bas, axis=1), lw=2, c='k', alpha=.7, picker=5)

    def show_fit(self, fit):
        self.axes.plot(np.sum(fit, axis=1), lw=2, c='r', alpha=.7, picker=5)

    def show_org(self, x):
        self.axes.plot(np.sum(x, axis=0), lw=2, c='b', alpha=.7, picker=5)

    def span_select_callback(self, xmin, xmax):
        cc = np.arange(0, self.x.shape[0])
        indmin, indmax = np.searchsorted(cc, (xmin, xmax))
        indmax = min(len(self.x)-1, indmax)
        self.segments.append((indmin, indmax))
        self.axes.vlines(xmin, self.oxy[1][0], self.oxy[1][1],
                         color='r', linestyles='--')
        self.axes.vlines(xmax, self.oxy[1][0], self.oxy[1][1],
                         color='r', linestyles='--')
        self.redraw()

    def updata_data(self, x):
        self.xx = x
        self.axes.clear()
        self.axes.set_xlabel("Scans")
        self.axes.set_ylabel("Instensity")
        self.axes.set_title("Least Square Fitting(2D)", fontsize=9)
        self.x = x['d']
        self.y = np.sum(self.x, axis=1)
        self.axes.plot(self.y, lw=1, c='b', alpha=.7, picker=5)
        diff_y = max(self.y)-min(self.y)
        self.axes.set_xlim(0, len(self.y))
        self.axes.set_ylim(0, max(self.y)*1.1)
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.plotorg = True
        self.redraw()

    def select_inter(self, segments):
        for i in range(0, len(segments)):
            indmin, indmax = segments[i]
            self.axes.vlines(self.x[indmin], self.oxy[1][0], self.oxy[1][1],
                             color='r', linestyles='--')
            self.axes.vlines(self.x[indmax], self.oxy[1][0], self.oxy[1][1],
                             color='r', linestyles='--')

    def backremv(self, seg):
        mn = np.shape(self.x)
        bak2 = np.zeros(mn)
        for i in range(0, mn[1]):
            tiab = []
            reg = []
            for j in range(0, len(seg)):
                tt = range(seg[j][0],seg[j][1])
                tiab.extend(self.x[tt, i])
                reg.extend(np.arange(seg[j][0], seg[j][1]))
            rm = reg - np.mean(reg)
            tm = tiab - np.mean(tiab)
            b = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
            s = np.mean(tiab)-np.dot(np.mean(reg), b)
            b_est = s+b*np.arange(mn[0])
            bak2[:, i] = self.x[:, i]-b_est
        bak = self.x-bak2
        self.yy = bak2
        return bak2, bak

    def accept(self):
        self.xx['d'] = self.yy
        self.close()

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = LSFQDialg()
    window.show()
    sys.exit(app.exec_())