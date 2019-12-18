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

# 融合图片，这是一个有双眼皮的人
merge_url = 'https://cdn.faceplusplus.com.cn/mc-official/scripts/demoScript/images/demo-pic114.jpg'
# 这是没有双眼皮的人
template_url = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981557927&di=484eae05e3ed0c0f4d30914862a012a0&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20181110%2F2063daae7ad94d3294d21fda4d604a6b.jpeg'

origin_img_res = requests.get(template_url)
# text一般用于返回的文本
# content的一般用于对返回的其他数据类型
img_data_str = origin_img_res.content  # file.content 是读取的远程文件的字节流
origin_img_np_array = np.frombuffer(img_data_str, np.uint8)
# print(origin_img_np_array)
origin_img = cv2.imdecode(origin_img_np_array, cv2.IMREAD_COLOR)
cv2.imshow("originImg", origin_img)

# 面部特征分析API https://console.faceplusplus.com.cn/documents/118131136
# return_imagereset 是否返回人脸矫正后图片。合法值为：0不返回 1返回 注：本参数默认值为 0
facepp_feature_res = api.facialfeatures(image_url=template_url)
# 提取结果中的关键点

points_list = facepp_feature_res.denselandmark

# 左-眼睛上线条
left_eye_up_choose = []
# 左-眉毛下线条条
left_eyebrow_down_choose = []
# 左-鼻子
left_nose_choose = []
# 左-脸
left_face_choose = []
# 右边
right_eye_up_choose = []
right_eyebrow_down_choose = []
right_nose_choose = []
right_face_choose = []

dst_img = origin_img.copy()
point_color = (0, 0, 255)  # BGR
for i in range(32):
    left_eye_up = points_list.left_eye_eyelid["left_eye_eyelid_" + str(i)]
    left_eyebrow_down = points_list.left_eyebrow["left_eyebrow_" + str(32 + i)]

    right_eye_up = points_list.right_eye_eyelid["right_eye_eyelid_" + str(i)]
    right_eyebrow_down = points_list.right_eyebrow["right_eyebrow_" + str(32 + i)]

    dst_img[left_eye_up.y, left_eye_up.x] = point_color
    dst_img[left_eyebrow_down.y, left_eyebrow_down.x] = point_color
    dst_img[right_eye_up.y, right_eye_up.x] = point_color
    dst_img[right_eyebrow_down.y, right_eyebrow_down.x] = point_color

    left_eye_up_choose.append([left_eye_up.x, left_eye_up.y])
    left_eyebrow_down_choose.append([left_eyebrow_down.x, left_eyebrow_down.y])
    right_eye_up_choose.append([right_eye_up.x, right_eye_up.y])
    right_eyebrow_down_choose.append([right_eyebrow_down.x, right_eyebrow_down.y])

for i in range(16):
    left_nose = points_list.nose["nose_left_" + str((15 - i))]
    left_face = points_list.face["face_contour_left_" + str(63 - i)]

    right_nose = points_list.nose["nose_right_" + str((15 - i))]
    right_face = points_list.face["face_contour_right_" + str(63 - i)]

    dst_img[left_nose.y, left_nose.x] = point_color
    dst_img[left_face.y, left_face.x] = point_color
    dst_img[right_nose.y, right_nose.x] = point_color
    dst_img[right_face.y, right_face.x] = point_color

    left_nose_choose.append([left_nose.x, left_nose.y])
    left_face_choose.append([left_face.x, left_face.y])
    right_nose_choose.append([right_nose.x, right_nose.y])
    right_face_choose.append([right_face.x, right_face.y])

cv2.imshow("points", dst_img)
# 抠图
left_choose = left_eye_up_choose + left_nose_choose + left_eyebrow_down_choose + left_face_choose
right_choose = right_eye_up_choose + right_nose_choose + right_eyebrow_down_choose + right_face_choose
# 创建一个掩码来抠图，需要扣的地方置为1，不扣的就使用 np.zeros 创建为 0
mask_array = np.zeros(origin_img.shape, dtype=np.uint8)
pts_left = np.array(left_choose)
pts_right = np.array(right_choose)
cv2.fillPoly(mask_array, [pts_left], color=(1, 1, 1))
cv2.fillPoly(mask_array, [pts_right], color=(1, 1, 1))
# 人脸融合：https://console.faceplusplus.com.cn/documents/20813963
# template_rectangle参数中的数据要通过人脸检测api来获取
mergeFace_res = api.mergeface(template_url=template_url, merge_url=merge_url)
# 从结果中把facepp返回的合成图base64拿到，然后转码成 string，提供给 cv2 使用
combine_img_data_str = base64.b64decode(mergeFace_res["result"])
# 装成 numpy 数组
combine_img_np_array = np.frombuffer(combine_img_data_str, np.uint8)
# 解码成 cv2 的图片格式
combine_img = cv2.imdecode(combine_img_np_array, cv2.IMREAD_COLOR)
# 结果图 = 原图 扣掉眼皮部分 + 合成图眼皮不服
# 原图眼皮部分扣掉 = 原图 - 原图*mask
# 合成图眼皮部分 = 合成图 * mask
res_res = origin_img - origin_img * mask_array + combine_img * mask_array
cv2.imshow('combine', combine_img)
cv2.imshow('combine_mask', combine_img * mask_array)
cv2.imshow('resres', res_res)
# # 等待按键输入
cv2.waitKey()
# # 关闭所有窗口，释放资源
cv2.destroyAllWindows()
