import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt


img_ori = cv2.imread('C:/koreaAI/project/aaa.png')
#img_ori = cv2.imread('C:/KoreaAI/Image/testimg3.jpg')

height, width, channel = img_ori.shape    

# ## 원본 사진을 그레이 색상으로 바꾼다.
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Car',gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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
plate_chars.append(result_chars)

if has_digit and len(result_chars) > longest_text:
    longest_idx = i

#plt.subplot(len(plate_imgs), 1, i+1)
#plt.imshow(img_result, cmap='gray')