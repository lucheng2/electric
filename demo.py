import os
import numpy as np
from glob import glob

import cv2
from matplotlib import pyplot as plt

file_name = glob("./images/" + "*jpg")
template = cv2.imread('template/tr.png', 0)
template2 = cv2.imread('template/tl.png', 0)
template3 = cv2.imread('template/tm.png', 0)
for file in file_name:
    image_c = cv2.imread(file)  # uncomment if dataset not downloaded
    image = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    (filepath, tempfilename) = os.path.split(file)
    (filename, extension) = os.path.splitext(tempfilename)

    image = cv2.medianBlur(image, 5)
    ret, image = cv2.threshold(image, 60, 155, cv2.THRESH_BINARY)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    w, h = template.shape[::-1]
    w2, h2 = template2.shape[::-1]
    w3, h3 = template3.shape[::-1]
    # All the 6 methods for comparison in a list

    # Apply template Matching
    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    res2 = cv2.matchTemplate(image, template2, cv2.TM_CCORR_NORMED)
    res3 = cv2.matchTemplate(image, template3, cv2.TM_CCORR_NORMED)
    threshold = 0.915
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # top_left = max_loc
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image_c, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(image_c, top_left, bottom_right, 255, 2)
    loc2 = np.where(res2 >= threshold)
    for pt in zip(*loc2[::-1]):
        cv2.rectangle(image_c, pt, (pt[0] + w2, pt[1] + h2), (0, 0, 255), 2)
    loc3 = np.where(res3 >= threshold)
    for pt in zip(*loc3[::-1]):
        cv2.rectangle(image_c, pt, (pt[0] + w3, pt[1] + h3), (0, 0, 255), 2)

    cv2.imwrite("./result/" + filename + ".jpg", image_c)
    # plt.savefig("./result/" + filename + ".jpg", image)
    # plt.close()

    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(image, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # # plt.suptitle(meth)
    # plt.show()
