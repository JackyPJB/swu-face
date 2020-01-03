#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       pengj
    @   date    :       2020/1/1 16:05
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""
import numpy as np
import cv2

__author__ = 'Max_Pengjb'
#
# img = cv2.imread('./imgResource/gaussion/bb_after.png')
# print(img.shape)
# # 核 越大，越模糊
# blur5 = cv2.GaussianBlur(img, (5, 5), 0)
# blur7 = cv2.GaussianBlur(img, (7, 7), 0)
# blur11 = cv2.GaussianBlur(img, (11, 11), 0)
# kernel = np.ones((5, 5), np.float32) / 25
# ff = cv2.filter2D(img, -1, kernel)
# cv2.imshow('origin', img)
# cv2.imshow('gaussion5', blur5)
# cv2.imshow('gaussion7', blur7)
# cv2.imshow('gaussion11', blur11)
# cv2.imshow('ff', ff)

# 计算不规则区域的形心
img_color1 = cv2.imread('./imgResource/gaussion/combine_mask.png')
img_eye = np.copy(img_color1)
img_origin = cv2.imread('./imgResource/gaussion/bb.jpg')
img2 = cv2.cvtColor(img_color1, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(img2, 0, 255, 0)
contours, _ = cv2.findContours(thresh, 3, 2)
cnt = contours[0]
print(type(cnt), cnt.shape, cnt)
# compute the center of the contour
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv2.circle(img2, (cX, cY), 7, (255, 255, 255), -1)
x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形

center = (x + w // 2, y + h // 2)
# center = (宽，高）
cv2.circle(img2, center, 2, (0, 255, 0), 2)
cv2.circle(img_color1, center, 2, (0, 255, 0), 2)
cv2.imshow("COCO detections", img2)
cv2.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
cropped = img_color1[y:y + h, x:x + w]  # 裁剪坐标为[y0:y1, x0:x1]
# rect = cv2.minAreaRect(cnt)  # 最小外接矩形
# box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
# cv2.drawContours(img_color1, [box], 0, (255, 0, 0), 2)
cv2.imshow("img_color1", img_color1)
cv2.imshow("cropped", cropped)
# seamlessClone
# mask , np.clip 设置矩阵的 最大 最小 范围
mask = np.clip(cropped, 0, 1) * 255
center2 = (x + w // 2, y + h // 2)
# center = (宽，高）
monochrome_transfer = cv2.seamlessClone(cropped, img_origin, mask, center2, cv2.MONOCHROME_TRANSFER)
cv2.imshow("monochrome_transfer", monochrome_transfer)

# # 等待按键输入
cv2.waitKey()
# # 关闭所有窗口，释放资源
cv2.destroyAllWindows()
