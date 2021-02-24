import sys
from PyQt5.QtWidgets import *   # 사용할 PyQt5 모듈  
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread , QTime
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont , QPalette
from PyQt5 import QtGui, QtCore, QtWidgets

import numpy as np # 수학과학 연산 모듈
import cv2 # openCV 모듈
import dlib # 이미지 처리 모듈
import pytesseract # 문자인식 
from time import sleep 
from datetime import datetime, timedelta

import pymysql



carList = [''] # 차 번호판을 저장할 리스트
carTime = [''] # 입차시각 저장할 리스트

carMenuList = [] # 등록된 차량 리스트

class VideoThread(QThread):    # 웹캠 이미지 재생
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    bSaveFlag = False

    def __init__(self):
        super().__init__()
        self._run_flag = True 

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)

        frameRate = int(cap.get(cv2.CAP_PROP_FPS))

        while self._run_flag: #  _run_flag 가 트루일동안 계속 반복
            ret, cv_img = cap.read()   # 비디오웹탬 이미지(cv_img)를 읽어옴
            if ret:
                self.change_pixmap_signal.emit(cv_img) # 읽어온 cv_img를  change_pixmap_signal로 뿌려줌

            key = cv2.waitKey(frameRate)

            if self.bSaveFlag:
                cv2.imwrite("./frame.jpg", cv_img) 
                self.bSaveFlag = False

        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("주차요금 계산기")
        self.setGeometry(100, 100, 670, 500)
        
        layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        btlayout = QHBoxLayout()

        layout2 = QVBoxLayout()

        
        carMenuTe =QTextEdit("버튼을 눌러주세요")

        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)
        

        self.btnStart = QPushButton("등록")
        self.btnStart.clicked.connect(self.onStartButtonClicked)
        #self.btnStop = QPushButton("시간")
        #self.btnStop.clicked.connect(self.onStopButtonClicked)
        self.btncough = QPushButton("정산")
        self.btncough.clicked.connect(self.onEndButtonClicked)
        self.btnMenu = QPushButton("등록된 차 메뉴")
        self.btnMenu.clicked.connect(lambda: self.onCarMenuButtonClicked(carMenuTe))
        

        self.lblCar = QLabel("차량 번호: ")
        self.lblTime = QLabel("입차 시각: ")
        self.lblTime2 = QLabel("출차 시각: ")
        self.lbParkingTime = QLabel("주차 기간: ")
        self.lbParkingPrice = QLabel("주차 요금: ")

        price_label = QLabel("주차요금 표")
        price_label.setAlignment(Qt.AlignCenter)
        font = price_label.font()
        font.setPointSize(20)
        price_label.setFont(font)

        price2_label = QLabel(" 10초당  : 200만원")
        price2_label.setAlignment(Qt.AlignCenter)
        font = price2_label.font()
        font.setPointSize(30)
        price2_label.setFont(font)

        
        layout1.addWidget(self.image_label)
        btlayout.addWidget(self.btnStart)
        btlayout.addWidget(self.btncough)
        layout1.addLayout(btlayout)

        layout1.addWidget(self.lblCar)
        layout1.addWidget(self.lblTime)
        layout1.addWidget(self.lblTime2)
        layout1.addWidget(self.lbParkingTime)
        layout1.addWidget(self.lbParkingPrice)

        
        layout2.addWidget(price_label)
        layout2.addWidget(price2_label)
        layout2.addWidget(self.btncough)
        layout2.addWidget(self.btnMenu)
        layout2.addWidget(carMenuTe)
        
        layout2.addStretch(2)

        layout.addLayout(layout1)
        layout.addLayout(layout2)
        self.setLayout(layout)    
        
        # create the video capture thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def onCarMenuButtonClicked(self,carMenuTe): # 저장된 자동차 목록 
        carMenuTe.clear() #
        for car in carList:
            carMenuTe.append(car)
        

    def insert_db(self,carnumb): # 입차 데이터 DB에 저장
        conn = pymysql.connect(host="localhost",user="root",password="1234",db="mydb1",charset="utf8")


        try:
            mydb = conn.cursor()
            sql = "insert into carlist(car_number, intime) values(%s, now())"
            mydb.execute(sql, carnumb)
            conn.commit()

        finally:
            conn.close()   

    def update_db(self,_time,_price, carnumb): # 출차데이터 DB에 업데이트
        conn = pymysql.connect(host="localhost",user="root",password="1234",db="mydb1",charset="utf8")
        try:
            mydb = conn.cursor()
            sql ="update carlist set outtime=now(), parkingTime=%s, parkingPrice=%s where car_number=%s and outtime is null"
            vals = (_time, _price, carnumb)
            mydb.execute(sql, vals )
            conn.commit()
        finally:
            conn.close()

    def onStartButtonClicked(self): # 입차한 차 번호판 인식 및 입차시간 저장
        #비디오를 이미지로 캡쳐한다.
        self.thread.bSaveFlag = True
        #WAit0.5
        sleep(0.5)
        #self.thread.save()
        startTime = datetime.now()
        str_stTime = str(startTime)
        self.lblTime.setText("입차 시각: " +  str_stTime)
        carTime.append(startTime)  # 주차등록 시간 배열에 추가
        
        #image_ori=cv2.imread("./frame.jpg", cv2.IMREAD_UNCHANGED)
        img_ori = cv2.imread('./frame.jpg',cv2.IMREAD_UNCHANGED)
#img_ori = cv2.imread('C:/KoreaAI/Image/testimg3.jpg')
        height, width, channel = img_ori.shape    
# ## 원본 사진을 그레이 색상으로 바꾼다.
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

        # 노이즈를 줄이기위해 가우시안 블러를 사용
        # blurring이란 이미지의 고주파 부분을 조금 더 자연스럽게 바꾸어줄 수 있는 방법이다.
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

        # 이미지의 Threshold를 적용하여 검은색과 흰색으로 이진화 한다.
        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9
        )
        # ## 컨투어(contour)란 동일한 색 또는 동일한 픽셀값(강도,intensity)을 가지고 있는 영역의 경계선 정보, 물체의 윤곽선, 외형을 파악하는데 사용된다
        #
        # 흑백 이미지에서 컨투어(윤곽선)을 찾는다.
        contours, _ = cv2.findContours(
            img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
       
        # Bounding Rectangle은 컨투어 라인을 둘러싸는 사각형을 그리는 방법이다. 

        # 빈 이미지를 검은색으로 만든다.
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)
        # 리스트를 만들고
        contours_dict = []
        # 컨투어의 사각형 범위를 찾아서 검은 비어있는 이미지에 사각형을 그린다.
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h),
                        color=(255, 255, 255), thickness=2)

            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        # 컨투어를 감싸는 사각형들만 나온다

        # 번호판 크기 상수 지정
        MIN_AREA = 80
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        # 가능한 컨투어들을 저장해 놓는곳
        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']  # 면적=가로*세로
            ratio = d['w'] / d['h']  # 배율=가로/세로
            # 번호판일 확률이 높은 컨투어들을 저장
            if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)

        # 비어있는 이미지 파일을 만들고
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        # 가능한 컨투어 사각형만을 그린다.
        for d in possible_contours:
            #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(
                d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

        MAX_DIAG_MULTIPLYER = 5  # 컨투어 사이의 길이가 5배 안에 있어야 한다.
        MAX_ANGLE_DIFF = 12.0  # 첫번째 컨투어와 두번째 컨투어 중심사이의 대각선 최대값
        MAX_AREA_DIFF = 0.5  #
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 3  # 3개 이상이 되어야 번호판이다.

        # 찾는 함수
        def find_chars(contour_list):
            matched_result_idx = []

            # d1 컨투어와 d2 컨투어를 비교하여 체크
            for d1 in contour_list:
                matched_contours_idx = []
                for d2 in contour_list:
                    if d1['idx'] == d2['idx']:
                        continue

                    dx = abs(d1['cx'] - d2['cx'])
                    dy = abs(d1['cy'] - d2['cy'])
                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                    # 컨투어 사이의 거리 구한다
                    distance = np.linalg.norm(
                        np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                    if dx == 0:
                        angle_diff = 90
                    else:
                        angle_diff = np.degrees(np.arctan(dy / dx))

                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] *
                                    d2['h']) / (d1['w'] * d1['h'])
                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours_idx.append(d2['idx'])

                # append this contour
                matched_contours_idx.append(d1['idx'])

                # 후보군의 갯수가 3보다 작다면 제외
                if len(matched_contours_idx) < MIN_N_MATCHED:
                    continue

                # 최종 후보군에 추가한다.
                matched_result_idx.append(matched_contours_idx)

                unmatched_contour_idx = []
                for d4 in contour_list:
                    if d4['idx'] not in matched_contours_idx:
                        unmatched_contour_idx.append(d4['idx'])

                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

                # 재귀함수 recursive하게 또 돌린다.
                recursive_contour_list = find_chars(unmatched_contour)

                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)

                break

            return matched_result_idx


        result_idx = find_chars(possible_contours)

        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))

        # visualize possible contours
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for r in matched_result:
            for d in r:
                #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(
                    d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)


        PLATE_WIDTH_PADDING = 1.3  # 1.3
        PLATE_HEIGHT_PADDING = 1.5  # 1.5
        MIN_PLATE_RATIO = 3
        MAX_PLATE_RATIO = 10

        plate_imgs = []
        plate_infos = []

        
        # 최종 result에 대해서 순서대로 정렬하고 center x , y 구하고
        for i, matched_chars in enumerate(matched_result):    
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) # 뒤죽박죽 순서를 정렬해주고

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2 # result의 처음인덱스 끝인덱스 중심 좌표 x
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2 # '' 중심 좌표 y 

            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]         # ' 평균길이
                        ['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING) # 평균 높이

            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy'] #  기울어진 번호판의  센터 높이
            triangle_hypotenus = np.linalg.norm(                                # 기울어진 번호판의 대각선길이
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )

            # 각도를 구하고
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))    # 아크탄젠트(높이 / 대각선) 해주고 세타

            # cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)  회전
            rotation_matrix = cv2.getRotationMatrix2D(
                center=(plate_cx, plate_cy), angle=angle, scale=1.0)

            # 이미지의 위치를 변경
            img_rotated = cv2.warpAffine(
                img_thresh, M=rotation_matrix, dsize=(width, height))

            # 번호판 부분만 crop 자른다.
            
            img_cropped = cv2.getRectSubPix(img_rotated, patchSize=(int(plate_width), int(plate_height)), center=(int(plate_cx), int(plate_cy)))
            
            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                continue

            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })

        plate_chars = []
        longest_idx, longest_text = -1, 0
        
        img_result = img_cropped
        #cv2.imwrite('C:/KoreaAI/Image/t1-er_plate.jpg', img_result)

        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7')

        result_chars1 = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars1 += c

        print(result_chars1)
        isStored = False
        for num in carList:
            if num ==result_chars1:
                QMessageBox.information(self,"경고","이미 주차된 번호 입니다.")
                isStored = True
                break  # 차들 번호판 저장

        if isStored == False:
            carList.append(result_chars1)
            self.lblCar.setText("차량 번호: " +  result_chars1)
            self.insert_db(result_chars1)
        
        
        

        self.lblTime2.setText("출차 시각: ")
        self.lbParkingTime.setText("주차 기간: ")
        self.lbParkingPrice.setText("주차 요금: ")
                
        


    def onEndButtonClicked(self): # 출차한 차 번호판 인식 및 시간,요금 저장
        #비디오를 이미지로 캡쳐한다.
        self.thread.bSaveFlag = True
        #WAit0.5
        sleep(0.5)
        #self.thread.save()

        EndTime = datetime.now()
        str_stTime = str(EndTime)
        self.lblTime2.setText("출차 시각: " +  str_stTime)
          # 주차등록 시간 배열에 추가
        

        #image_ori=cv2.imread("./frame.jpg", cv2.IMREAD_UNCHANGED)
        img_ori = cv2.imread('./frame.jpg',cv2.IMREAD_UNCHANGED)
#img_ori = cv2.imread('C:/KoreaAI/Image/testimg3.jpg')

        height, width, channel = img_ori.shape 

# ## 원본 사진을 그레이 색상으로 바꾼다.
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

        # 노이즈를 줄이기위해 가우시안 블러를 사용
        # blurring이란 이미지의 고주파 부분을 조금 더 자연스럽게 바꾸어줄 수 있는 방법이다.
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

        # 이미지의 Threshold를 적용하여 검은색과 흰색으로 이진화 한다.
        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9
        )

        # ## 컨투어(contour)란 동일한 색 또는 동일한 픽셀값(강도,intensity)을 가지고 있는 영역의 경계선 정보, 물체의 윤곽선, 외형을 파악하는데 사용된다
        #
        # 흑백 이미지에서 컨투어(윤곽선)을 찾는다.
        contours, _ = cv2.findContours(
            img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        # 빈 이미지를 만들고
        #temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        # 빈 이미지에 전체 컨투어를 그린다.
        #cv2.drawContours(temp_result, contours=contours,
        #                contourIdx=-1, color=(255, 255, 255))

        # Bounding Rectangle은 컨투어 라인을 둘러싸는 사각형을 그리는 방법이다. 

        # 빈 이미지를 검은색으로 만든다.
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        # 리스트를 만들고
        contours_dict = []

        # 컨투어의 사각형 범위를 찾아서 검은 비어있는 이미지에 사각형을 그린다.
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h),
                        color=(255, 255, 255), thickness=2)

            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        # 컨투어를 감싸는 사각형들만 나온다

        # 번호판 크기 상수 지정
        MIN_AREA = 80
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        # 가능한 컨투어들을 저장해 놓는곳
        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']  # 면적=가로*세로
            ratio = d['w'] / d['h']  # 배율=가로/세로
            # 번호판일 확률이 높은 컨투어들을 저장
            if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)

        # 비어있는 이미지 파일을 만들고
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        # 가능한 컨투어 사각형만을 그린다.
        for d in possible_contours:
            #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(
                d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

        MAX_DIAG_MULTIPLYER = 5  # 컨투어 사이의 길이가 5배 안에 있어야 한다.
        MAX_ANGLE_DIFF = 12.0  # 첫번째 컨투어와 두번째 컨투어 중심사이의 대각선 최대값
        MAX_AREA_DIFF = 0.5  #
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 3  # 3개 이상이 되어야 번호판이다.

        # 찾는 함수
        def find_chars(contour_list):
            matched_result_idx = []

            # d1 컨투어와 d2 컨투어를 비교하여 체크
            for d1 in contour_list:
                matched_contours_idx = []
                for d2 in contour_list:
                    if d1['idx'] == d2['idx']:
                        continue

                    dx = abs(d1['cx'] - d2['cx'])
                    dy = abs(d1['cy'] - d2['cy'])
                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                    # 컨투어 사이의 거리 구한다
                    distance = np.linalg.norm(
                        np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                    if dx == 0:
                        angle_diff = 90
                    else:
                        angle_diff = np.degrees(np.arctan(dy / dx))

                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] *
                                    d2['h']) / (d1['w'] * d1['h'])
                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours_idx.append(d2['idx'])

                # append this contour
                matched_contours_idx.append(d1['idx'])

                # 후보군의 갯수가 3보다 작다면 제외
                if len(matched_contours_idx) < MIN_N_MATCHED:
                    continue

                # 최종 후보군에 추가한다.
                matched_result_idx.append(matched_contours_idx)

                unmatched_contour_idx = []
                for d4 in contour_list:
                    if d4['idx'] not in matched_contours_idx:
                        unmatched_contour_idx.append(d4['idx'])

                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

                # 재귀함수 recursive하게 또 돌린다.
                recursive_contour_list = find_chars(unmatched_contour)

                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)

                break

            return matched_result_idx


        result_idx = find_chars(possible_contours)

        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))

        # visualize possible contours
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for r in matched_result:
            for d in r:
                #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(
                    d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)


        PLATE_WIDTH_PADDING = 1.3  # 1.3
        PLATE_HEIGHT_PADDING = 1.5  # 1.5
        MIN_PLATE_RATIO = 3
        MAX_PLATE_RATIO = 10

        plate_imgs = []
        plate_infos = []

        # 최종 result에 대해서 순서대로 정렬하고 center x , y 구하고
        for i, matched_chars in enumerate(matched_result):    
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) # 뒤죽박죽 순서를 정렬해주고

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2 # result의 처음인덱스 끝인덱스 중심 좌표 x
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2 # '' 중심 좌표 y 

            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]         # ' 평균길이
                        ['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING) # 평균 높이

            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy'] #  기울어진 번호판의  센터 높이
            triangle_hypotenus = np.linalg.norm(                                # 기울어진 번호판의 대각선길이
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )

            # 각도를 구하고
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))    # 아크탄젠트(높이 / 대각선) 해주고 세타

            # cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)  회전
            rotation_matrix = cv2.getRotationMatrix2D(
                center=(plate_cx, plate_cy), angle=angle, scale=1.0)

            # 이미지의 위치를 변경
            img_rotated = cv2.warpAffine(
                img_thresh, M=rotation_matrix, dsize=(width, height))

            # 번호판 부분만 crop 자른다.
            img_cropped = cv2.getRectSubPix(img_rotated, patchSize=(
                int(plate_width), int(plate_height)), center=(int(plate_cx), int(plate_cy)))

            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                continue

            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })

        plate_chars = []
        longest_idx, longest_text = -1, 0

        img_result = img_cropped
        
        #cv2.imwrite('C:/KoreaAI/Image/t1-er_plate.jpg', img_result)
        
        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
        
        print(result_chars)
        self.lblCar.setText("차량 번호: " +  result_chars)
        try:
            carindex = carList.index(result_chars)
            print(carindex)
        except ValueError:
            print("wwwww")

        try:
            if (carList[carindex]==result_chars):
                extime = (EndTime-carTime[carindex]).seconds
                print(extime)
                str_exTime = str(extime)
                self.lbParkingTime.setText("주차 기간: "+str_exTime+ " 초")
                price = (extime/10)*2000000
                #final_price = round(price)
                str_Price = str(price)
                self.lbParkingPrice.setText("주차 요금: "+str_Price+ " 원")
                carList.remove(carList[carindex]) # 출차된 번호판 삭제
                carTime.remove(carTime[carindex]) # 출차된 차량 시간 삭제

                self.update_db(str_exTime,str_Price, result_chars)    
                
        except UnboundLocalError:
            QMessageBox.information(self,"경고","없는 주차번호 입니다.")
            print("번호가 없습니다.")
        

        
        

    def closeEvent(self, event): # 창을 종료하면 쓰레드가 멈춰 stop()에  false 값을 보냄
        self.thread.stop()
        event.accept()
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img): # 이미지를  화면에 나타냄
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):    # cv로 읽어온 이미지를 Qt이미지로 변환 
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    sys.exit(app.exec_())