#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       pengj
    @   date    :       2019/12/2 14:38
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :
-------------------------------------------------
"""
import base64

import requests
import cv2
import numpy as np
# import PythonSDK
from PythonSDK.facepp import API

# 初始化对象，进行api的调用工作
api = API()

__author__ = 'Max_Pengjb'

# 这是没有双眼皮的人
# template_url = 'https://i.loli.net/2019/12/31/QGaV9YOPmMHle13.jpg'
# 通过 requests 获取图床上的图片
# origin_img_res = requests.get(template_url)
# text一般用于返回的文本
# content的一般用于对返回的其他数据类型
# img_data_str = origin_img_res.content  # file.content 是读取的远程文件的字节流
# origin_img_np_array = np.frombuffer(img_data_str, np.uint8)
# print(origin_img_np_array)
# origin_img = cv2.imdecode(origin_img_np_array, cv2.IMREAD_COLOR)
origin_img = cv2.imread('./imgResource/gaussion/bb.jpg')
left_eye = './imgResource/gaussion/left.jpg'
right_eye = './imgResource/gaussion/right.jpg'

# 在原图中随便找个点，作为起始位置
img_h, img_w, _ = origin_img.shape
print(origin_img.shape)
default_center = [img_w // 2, img_h // 2]


# 把眼皮的外接矩形抠出来,没有经过这个函数处理过的图片，最好先处理一下（从ps6中来，透明背景是 255，这里要把他变成 0）
def eye_rectangle(img_url):
    img_eye = cv2.imread(img_url)
    # 原来是白背景把他变成黑背景
    img_eye[img_eye == 255] = 0
    # 注意：opencv 中有求外接矩阵的算法 cv2.boundingRect(image) 或者 cv2.boundingRect(contour),
    # 只是这个 image 还需要通过二值图需要转成二值图或者灰度图(cv2.cvtColor(merge_obj_left_eye, cv2.COLOR_BGR2GRAY)
    eye_gray = cv2.cvtColor(img_eye, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(eye_gray, 0, 255, 0)
    # contours, _ = cv2.findContours(thresh, 3, 2)
    # contour = contours[0]
    # x1, y1, w1, h1 = cv2.boundingRect(contour)  # 外接矩形
    _x, _y, _w, _h = cv2.boundingRect(eye_gray)  # 外接矩形

    # 先用 mask 找出左右眼皮，然后根据外框矩形，把那一块抠出来
    # 把合成后的左眼右眼抠图出来
    # Read images : src image will be cloned into dst
    return img_eye[_y:_y + _h, _x:_x + _w]


'''
对于cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)来讲：
obj代表的是子图，由cv2读进来的数组文件；
im代表的是母图，也是由cv2都进来的数组文件；
mask代表掩模，因为你并不需要把子图所有的部分都贴进来，所以可以用mask划分出一个兴趣域。只需要用0和255区分就可以。如果你不想管这个mask，直接都设置成255就行了；
center表示坐标，你打算在母图的哪个位置放子图。这里是放在中间。
cv2.NORMAL_CLONE代表融合的模式，可以比较 cv2.NORMAL_CLONE和cv2.MIXED_CLONE的差别。、
'''
cropped_left = eye_rectangle(left_eye)
cropped_right = eye_rectangle(right_eye)
# cv2.imshow('cropped_left', cropped_left)
# np.clip 设置矩阵的数值 最大 最小 范围
mask_left = np.clip(cropped_left, 0, 1) * 255
mask_right = np.clip(cropped_right, 0, 1) * 255


# 输入裁剪的图，和需要放大/缩小的（宽，高）倍数
def resize_cropped_mask(input_img, w, h):
    crop_img = cv2.resize(input_img, (0, 0), fx=w, fy=h, interpolation=cv2.INTER_NEAREST)
    crop_mask = np.clip(crop_img, 0, 1) * 255
    return crop_img, crop_mask


# 编辑眼睛 # 泊松融合，先左眼，再把右眼加进去
def add_eye(src_cropped, src_img, center, w=1, h=1):
    # w=1,h=1 就是初始，不缩放
    cropped, mask = resize_cropped_mask(src_cropped, w, h)
    monochrome_tmp = cv2.seamlessClone(cropped, src_img, mask, tuple(center), cv2.MONOCHROME_TRANSFER)
    cv2.imshow('res', monochrome_tmp)
    flag = True
    while flag:
        kk = cv2.waitKeyEx(0)
        print(type(kk), kk)
        # left = 2424832
        if kk == 2424832:
            center[0] -= 1
        # right = 2555904
        elif kk == 2555904:
            center[0] += 1
        # up = 2490369
        elif kk == 2490368:
            center[1] -= 1
        # down = 2621440
        elif kk == 2621440:
            center[1] += 1
        elif kk in [56, 50, 54, 52, 43, 45]:
            # 高度缩放 扩大 数字 ’8‘ = 56
            if kk == 56:
                h += 0.01
            # 高度缩放 缩小 数字 ’2‘ = 50
            elif kk == 50:
                h -= 0.01
            # 宽度缩放 扩大 数字 ’6‘ = 54
            elif kk == 54:
                w += 0.01
            # 宽度缩放 缩小 数字 ’4‘ = 52
            elif kk == 52:
                w -= 0.01
            # 整体 扩大 ’+‘ = 43
            elif kk == 43:
                w += 0.01
                h += 0.01
            # 整体 缩小 ’-‘ = 45
            elif kk == 45:
                w -= 0.01
                h -= 0.01
            cropped, mask = resize_cropped_mask(src_cropped, w, h)
        # 确定或者退出
        elif kk in [13, 27, 32, -1]:  # 13 回车 27 EC 32 空格
            flag = False
        monochrome_tmp = cv2.seamlessClone(cropped, src_img, mask, tuple(center), cv2.MONOCHROME_TRANSFER)
        cv2.imshow('res', monochrome_tmp)
    return monochrome_tmp


# 加左眼
print(default_center)
monochrome_left_res = add_eye(cropped_left, origin_img, default_center)
print(default_center)
# 加右眼
monochrome_res = add_eye(cropped_right, monochrome_left_res, default_center)
# Write results
# cv2.imwrite("images/opencv-normal-clone-example.jpg", normal_clone)
# cv2.imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone)
# 等待按一个按钮关闭
cv2.waitKey(0)
# 关闭所有窗口，释放资源
cv2.destroyAllWindows()
