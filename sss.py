import cv2
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt ,QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont

import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import datetime


class ShowVideo(QtCore.QObject):

    flag = 0

    camera = cv2.VideoCapture(0)
    
    ret, image = camera.read()
       

    height, width = image.shape[:2]
    
    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        global image

        run_video = True
        while run_video:
            ret, image = self.camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                    self.width,
                                    self.height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)


            if self.flag:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                

               

                


            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()

   

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # 앱 객체 생성

    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()
    #image_viewer2 = ImageViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    #vid.VideoSignal2.connect(image_viewer2.setImage)
    
    #push_button1 = QtWidgets.QPushButton('Start')
    #push_button2 = QtWidgets.QPushButton('Canny')
    #push_button1.clicked.connect(vid.startVideo)
    #push_button2.clicked.connect(vid.canny)

    push_button2 = QtWidgets.QPushButton('등록')
    push_button3 = QtWidgets.QPushButton('정산')
    push_button4 = QtWidgets.QPushButton('등록된 차량메뉴')

    label1 = QLabel('차량번호 주차요금')
    label1.setAlignment(Qt.AlignCenter)
    font1 = label1.font()
    font1.setPointSize(10)
    label1.setFont(font1)

    label2 = QLabel('주차요금 표')
    label2.setAlignment(Qt.AlignCenter)
    font2 = label2.font()
    font2.setPointSize(20)
    label2.setFont(font2)

    label3 = QLabel('시간당 : 200만원')
    label3.setAlignment(Qt.AlignCenter)
    font3 = label3.font()
    font3.setPointSize(50)
    
    label3.setFont(font3)


#------------------------------------------------------------

    vertical_layout1 = QtWidgets.QVBoxLayout()
    vertical_layout2 = QtWidgets.QVBoxLayout()

    button_layout = QtWidgets.QHBoxLayout()

    horizontal_layout = QtWidgets.QHBoxLayout()
    #horizontal_layout.addWidget(image_viewer2)


    vertical_layout1.addWidget(image_viewer1)   # 캠화면
    
    #vertical_layout1.addWidget(push_button1)    # 버튼
    vertical_layout1.addWidget(label1)          # 텍스트  
    #vertical_layout.addWidget(push_button2)
#------------------------------------------------------좌측 반쪽 화면




    vertical_layout2.addWidget(label2)       # 주차요금 텍스트
    vertical_layout2.addStretch(1)
    vertical_layout2.addWidget(label3)      # 주차요금표

    button_layout.addWidget(push_button2)    #등록 버튼
    button_layout.addWidget(push_button3)    # 정산 버튼 --- 수평 합체
    vertical_layout2.addStretch(2)
    vertical_layout2.addLayout(button_layout)    #  수평합체한 버튼 
    vertical_layout2.addWidget(push_button4)     #  등록됭 차량 메뉴
    vertical_layout2.addStretch(1)
#-------------------------------------------------------우측 반쪽 화면



    horizontal_layout.addLayout(vertical_layout1)
    horizontal_layout.addLayout(vertical_layout2) # 좌측화면 우측화면 수평 합체
    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(horizontal_layout)
    layout_widget.show()
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    vid.startVideo()
    sys.exit(app.exec_())

