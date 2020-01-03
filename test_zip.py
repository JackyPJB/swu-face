#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       pengj
    @   date    :       2020/1/2 16:53
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""
import cv2
import numpy as np

__author__ = 'Max_Pengjb'

a = [[1, 2], [1, 3], [1, 4], [1, 5], [4, 2], [7, 2], [4, 2], [5, 2], [8, 2], [1, 9], [10, 2]]
# x, y = zip(*a)
# print(x, y)
# pts = np.array(a)
# print(type(pts), pts.shape, pts)
img = cv2.imread('./imgResource/gaussion/combine_mask.png', 0)

x1, y1, w1, h1 = cv2.boundingRect(img)  # 外接矩形
print(x1, y1, w1, h1)
scoped = img[y1:y1 + h1, x1:x1 + w1]
cv2.imshow('haha', scoped)

# # 等待按键输入
cv2.waitKey()
# # 关闭所有窗口，释放资源
cv2.destroyAllWindows()
