__author__ = 'Administrator'

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector, Button
import numpy as np

class TICPlot(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.ncr = dict()
        self.fileno = 0
        self.segments = []
        self.create_canvas()

    def create_canvas(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = plt.subplot(111)
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.axes.tick_params(axis='both', labelsize=8)

        plt.subplots_adjust(bottom=0.25, top=0.90, left=0.08, right=0.9)
        self.zoom = RectangleSelector(self.axes, self.rectangle_select_callback,
                                 drawtype='box', useblit=True,
                                 button=[1], minspanx=5,
                                 minspany=5,
                                 spancoords='pixels')
        self.span = SpanSelector(self.axes, self.span_select_callback, 'horizontal', minspan=0.002,
                                 useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'),
                                 onmove_callback=None,
                                 button=[1])
        axbasic = plt.axes([0.59, 0.04, 0.08, 0.075])
        axPan = plt.axes([0.7, 0.04, 0.08, 0.075])
        axPick = plt.axes([0.81, 0.04, 0.08, 0.075])
        self.btnBasic = Button(axbasic, 'Zoom')
        self.btnPan = Button(axPan, 'Span')
        self.btnPick = Button(axPick, 'Undo')
        axtic = plt.axes([0.92, 0.825, 0.06, 0.075])
        axext = plt.axes([0.92, 0.725, 0.06, 0.075])
        axres = plt.axes([0.92, 0.625, 0.06, 0.075])
        self.TICbutton = Button(axtic, 'TIC')
        self.EXTbutton = Button(axext, 'EXT')
        self.RESbutton = Button(axres, 'RES')

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

        self.btnBasic.on_clicked(self.zoom_mode)
        self.btnPan.on_clicked(self.span_mode)
        self.btnPick.on_clicked(self.undo_mode)

        self.TICbutton.on_clicked(self.slot_tic)
        self.EXTbutton.on_clicked(self.slot_ext)
        self.RESbutton.on_clicked(self.slot_res)

        self.zoom_mode(True)
        self.redraw()
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]

    def slot_tic(self,event):
        self.add_tic(self.ncr, self.fileno)

    def slot_ext(self, event):
        self.emit(SIGNAL("ext_plot"))

    def slot_res(self, event):
        self.emit(SIGNAL("res_plot"), )

    def add_tic(self, ncr, fn):
        if not len(self.segments):
            self.fn = fn
            self.ncr = ncr
            tic = self.ncr.tic()
            self.x = tic['rt']
            self.y = tic['val']
            self.axes.clear()
            self.axes.set_xlabel("Retention Time")
            self.axes.set_ylabel("Instensity")
            self.axes.set_title(str(fn),fontsize=9)
            self.axes.plot(self.x, self.y, lw=1, c='b', alpha=.7, picker=5)
            self.axes.set_xlim(min(self.x), max(self.x))
            self.axes.set_ylim(min(self.y), max(self.y)*1.1)
            ymino, ymaxo = self.axes.get_ylim()
            xmino, xmaxo = self.axes.get_xlim()
            self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
            self.redraw()

    # def add_ext(self):

    def clear_data(self):
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.redraw()

    def redraw(self):
        self.canvas.draw()
        self.update()

    def zoom_mode(self, event):
        self.zoom.set_active(True)
        self.span.set_active(False)
        self.cidPress = self.canvas.mpl_connect('button_press_event', self.mouse_press_callback)
        self.cidRelease = self.canvas.mpl_connect('button_release_event', self.mouse_release_callback)
        self.span.disconnect_events()
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw()
        print "zoom"

    def span_mode(self, event):
        self.zoom.set_active(False)
        self.span.set_active(True)
        self.span.connect_event('motion_notify_event', self.span.onmove)
        self.span.connect_event('button_press_event', self.span.press)
        self.span.connect_event('button_release_event', self.span.release)
        self.span.connect_event('draw_event', self.span.update_background)
        # self.span.connect_event('button_press_event', self.mouse_press_callback)
        self.rightdblclick = False
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw()
        print "span"

    def undo_mode(self, event):
        self.zoom.set_active(True)
        self.span.disconnect_events()
        self.cidPress = self.canvas.mpl_connect('button_press_event', self.mouse_press_callback)
        if len(self.segments) >= 1:
            self.emit(SIGNAL("delete_SELECT"), self.segments[-1])
            del self.axes.collections[:]
            self.segments = self.segments[:-1]
            self.select_inter(self.segments)
            if self.ind_right_press != 0:
                self.axes.vlines(self.xdata, self.oxy[1][0], self.oxy[1][1],
                                                  color='g', linestyles='-')
            self.redraw()
        print "undo"

    def mouse_press_callback(self, event):
        if (event.button == 1 and event.dblclick == True):
            self.leftdblclick = True
        if event.button == 3:
            self.xdata = event.xdata
            self.ind_right_press = np.searchsorted(self.x, event.xdata)
            del self.axes.collections[:]
            self.axes.vlines(event.xdata, self.oxy[1][0], self.oxy[1][1],
                             color='g', linestyles='-')
            self.select_inter(self.segments)
            self.redraw()
            self.emit(SIGNAL("MASS_SELECT"), self.ncr, self.ind_right_press)

    def select_inter(self, segments):
        for i in range(0, len(segments)):
            indmin, indmax = segments[i]
            self.axes.vlines(self.x[indmin], self.oxy[1][0], self.oxy[1][1],
                             color='r', linestyles='--')
            self.axes.vlines(self.x[indmax], self.oxy[1][0], self.oxy[1][1],
                             color='r', linestyles='--')

    def mouse_release_callback(self, event):
        if (self.leftdblclick):
            self.leftdblclick = False
            self.axes.set_xlim(self.oxy[0])
            self.axes.set_ylim(self.oxy[1])
            self.redraw()
        if (self.rightdblclick):
            self.rightdblclick = False
            del self.axes.collections[:]
            self.redraw()

    def rectangle_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if eclick.button == 1 and erelease.button == 1:
            self.axes.set_xlim(min(x1, x2), max(x1, x2))
            self.axes.set_ylim(min(y1, y2), max(y1, y2))
            self.redraw()

    def span_select_callback(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.x, (xmin, xmax))
        indmax = min(len(self.x)-1, indmax)
        self.segments.append((indmin, indmax))
        self.axes.vlines(xmin, self.oxy[1][0], self.oxy[1][1],
                         color='r', linestyles='--')
        self.axes.vlines(xmax, self.oxy[1][0], self.oxy[1][1],
                         color='r', linestyles='--')
        self.redraw()
        self.emit(SIGNAL("range_SELECT"), self.ncr, (indmin, indmax))

    def loading(self):
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.redraw()
        self.zoom_mode(True)
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = TICPlot()
    window.show()
    sys.exit(app.exec_())

