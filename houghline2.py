import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/11.png')
print(img.dtype)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 150, 300, apertureSize=3)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

minLineLength = 1000
maxLineGap = 5
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.imwrite('result/houghlines4.png', img)
