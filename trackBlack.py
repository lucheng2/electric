import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/11.png')
# 转换到 HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 设定蓝色的阈值
lower_blue = np.array([0, 100, 100])
upper_blue = np.array([150, 250, 250])
# 根据阈值构建掩模
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# 对原图像和掩模进行位运算
res = cv2.bitwise_and(img, img, mask=mask)

# plt.subplot(121)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(121)
plt.imshow(mask, cmap='gray')
plt.title('mask Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(res, cmap='gray')
plt.title('result Image'), plt.xticks([]), plt.yticks([])
plt.show()

# # 显示图像
# cv2.imshow('img', img)
# cv2.imshow('mask', mask)
# cv2.imshow('res', res)
# k = cv2.waitKey(5) & 0xFF
# # 关闭窗口
# cv2.destroyAllWindows()
