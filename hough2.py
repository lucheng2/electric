import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("images/9.jpg", cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 1.矩形边界框检测
    # (1)计算出一个简单的边框
    x, y, w, h = cv2.boundingRect(c)

    # (2)将轮廓转换为(x,y)坐标，加上矩形的高度和宽度，绘制矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 2.最小矩形区域检测
    # (1)计算出包围目标的最小区域
    rect = cv2.minAreaRect(c)

    # (2)计算最小面积矩形的坐标
    box = cv2.boxPoints(rect)

    # (3)坐标归一化为整型
    box = np.int0(box)

    # (4)绘制轮廓
    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

    # 3.最小闭圆检测
    # (1)计算最小闭圆的中心和半径
    # (x, y), radius = cv2.minEnclosingCircle(c)

    # (2)坐标归一化为整型
    # center = (int(x), int(y))
    # radius = int(radius)

    # (3)绘制圆
    # img = cv2.circle(img, center, radius, (255, 0, 0), 2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 2)  # 绘制边沿轮廓
cv2.imshow("contours", img)

cv2.waitKey()
cv2.destroyAllWindows()