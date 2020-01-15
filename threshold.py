import cv2
import numpy as np
from matplotlib import pyplot as plt


imgo = cv2.imread('template/switch2.png')
img = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
# 中值滤波
img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 60, 155, cv2.THRESH_BINARY)
# 11 为 Block size, 2 为 C 值
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(th2, -1, kernel)
dst2 = cv2.filter2D(th3, -1, kernel)

th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
dst2 = cv2.morphologyEx(dst2, cv2.MORPH_OPEN, kernel)

# ret, thresh = cv2.threshold(dst2, 127, 255, cv2.THRESH_BINARY)
# contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for c in contours:
#     # 1.矩形边界框检测
#     # (1)计算出一个简单的边框
#     x, y, w, h = cv2.boundingRect(c)
#
#     # (2)将轮廓转换为(x,y)坐标，加上矩形的高度和宽度，绘制矩形
#     cv2.rectangle(imgo, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv2.rectangle(dst2, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     # 2.最小矩形区域检测
#     # (1)计算出包围目标的最小区域
#     rect = cv2.minAreaRect(c)
#
#     # (2)计算最小面积矩形的坐标
#     box = cv2.boxPoints(rect)
#
#     # (3)坐标归一化为整型
#     box = np.int0(box)
#
#     # (4)绘制轮廓
#     cv2.drawContours(imgo, [box], 0, (0, 255, 0), 3)
#     cv2.drawContours(dst2, [box], 0, (0, 255, 0), 3)

    # 3.最小闭圆检测
    # (1)计算最小闭圆的中心和半径
    # (x, y), radius = cv2.minEnclosingCircle(c)
    #
    # # (2)坐标归一化为整型
    # center = (int(x), int(y))
    # radius = int(radius)
    #
    # # (3)绘制圆
    # img = cv2.circle(imgo, center, radius, (255, 0, 0), 2)

# cv2.drawContours(img, contours, -1, (255, 0, 0), 2)  # 绘制边沿轮廓
# cv2.imshow("contours", img)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'conv1', 'edges']
images = [imgo, th1, th2, th3, dst, dst2]
for i in range(6):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite("./template/t1.png", dst)
cv2.imwrite("./template/t2.png", dst2)
cv2.imwrite("./template/t3.png", th1)
cv2.imwrite("./template/t4.png", th2)
cv2.imwrite("./template/t5.png", th3)
