__author__ = 'Administrator'

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector, Button
import numpy as np

class REFPlotWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.handles = []
        self.create_canvas()
        self.segments = []

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
                                 button=[1],
                                 minspanx=5, minspany=5,
                                 spancoords='pixels')
        self.span = SpanSelector(self.axes, self.span_select_callback, 'horizontal', minspan=0.002, useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'),
                                 onmove_callback=None,
                                 button=[1])
        axbasic = plt.axes([0.59, 0.04, 0.08, 0.075])
        axPan = plt.axes([0.7, 0.04, 0.08, 0.075])
        axPick = plt.axes([0.81, 0.04, 0.08, 0.075])
        self.btnBasic = Button(axbasic, 'Zoom')
        self.btnPan = Button(axPan, 'Span')
        self.btnPick = Button(axPick, 'Undo')

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

        self.btnBasic.on_clicked(self.zoom_mode)
        self.btnPan.on_clicked(self.span_mode)
        self.btnPick.on_clicked(self.undo_mode)

        # self.btnBasic.clicked.connect(self.zoom_mode)
        # self.btnPan.clicked.connect(self.span_mode)
        # self.btnPick.clicked.connect(self.undo_mode)

        self.zoom_mode(True)
        self.redraw()
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]

    def redraw(self):
        self.canvas.draw()
        self.update()

    def zoom_mode(self, event):
        self.zoom.set_active(True)
        self.cidPress = self.canvas.mpl_connect('button_press_event', self.mouse_press_callback)
        self.cidRelease = self.canvas.mpl_connect('button_release_event', self.mouse_release_callback)
        self.span.disconnect_events()
        self.leftdblclick = False
        self.rightdblclick = False
        self.redraw()
        print "zoom"

    def span_mode(self, event):
        self.zoom.set_active(False)
        self.span.connect_event('motion_notify_event', self.span.onmove)
        self.span.connect_event('button_press_event', self.span.press)
        self.span.connect_event('button_release_event', self.span.release)
        self.span.connect_event('draw_event', self.span.update_background)
        self.span.connect_event('button_press_event', self.mouse_press_callback)
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
            # del self.axes.collections[:]
            self.axes.set_xlim(min(x1, x2), max(x1, x2))
            self.axes.set_ylim(min(y1, y2), max(y1, y2))
            self.redraw()

    def span_select_callback(self, xmin, xmax):
        imin, imax = np.searchsorted(self.x, (xmin, xmax))
        imax = min(len(self.x)-1, imax)
        seg = self.x[[imin, imax]]
        self.segments.append((seg[0], seg[1]))
        self.axes.scatter(xmin, self.y[imin], s=30, marker='^',
            color='red', label='x1 samples')
        self.axes.scatter(xmax, self.y[imax], s=30, marker='v',
            color='red', label='x1 samples')

        # self.axes.vlines(xmin, self.oxy[1][0], self.oxy[1][1],
        #                  color='r', linestyles='--')
        # self.axes.vlines(xmax, self.oxy[1][0], self.oxy[1][1],
        #                  color='r', linestyles='--')
        self.redraw()
        self.emit(SIGNAL("range_SELECT"), (seg[0], seg[1]))

    def add_tic(self, ncr, fn):
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.axes.set_title(str(fn), fontsize=9)
        self.ncr = ncr
        tic = self.ncr.tic()
        self.x = tic['rt']
        self.y = tic['val']
        self.ind_right_press = 0
        self.axes.plot(self.x, self.y, lw=1, c='b', alpha=.7, picker=5)
        self.axes.plot(self.x, np.zeros(self.y.shape[0]), 'k--')
        diff_y = max(self.y)-min(self.y)
        self.axes.set_xlim(min(self.x), max(self.x))
        self.axes.set_ylim(min(self.y)-diff_y/50, max(self.y)*1.1)
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy = [(xmino, xmaxo), (ymino, ymaxo)]
        self.redraw()

    def update_segments(self, delete_segs):
        for delete_seg in delete_segs:
            for index, seg in enumerate(self.segments):
                if delete_seg == seg:
                    delete_no = index
                    break
            self.segments.remove(self.segments[delete_no])
        del self.axes.collections[:]
        self.select_inter(self.segments)
        self.redraw()

    def select_inter(self, segments):
        for i in range(0, len(segments)):
            xmin, xmax = segments[i]
            imin, imax = np.searchsorted(self.x, (xmin, xmax))
            self.axes.scatter(xmin, self.y[imin], s=30, marker='^',
                color='red', label='x1 samples')
            self.axes.scatter(xmax, self.y[imax], s=30, marker='v',
                color='red', label='x1 samples')

    def clear_data(self):
        self.segments = []
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.redraw()
        self.zoom_mode(True)

    def loading(self, seg, ncr, fn):
        self.segments = []
        self.axes.clear()
        self.axes.set_xlabel("Retention Time")
        self.axes.set_ylabel("Instensity")
        self.redraw()
        self.zoom_mode(True)
        ymino, ymaxo = self.axes.get_ylim()
        xmino, xmaxo = self.axes.get_xlim()
        self.oxy=[(xmino, xmaxo), (ymino, ymaxo)]

        self.add_tic(ncr, fn)
        self.segments = seg
        self.select_inter(self.segments)
        self.redraw()

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = REFPlotWidget()
    window.show()
    sys.exit(app.exec_())
