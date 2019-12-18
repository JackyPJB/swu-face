import cv2
import numpy as np


def circle_fitness_demo():
    # 创建图像， 绘制初始点
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    x = np.array([30, 50, 100, 120])
    y = np.array([100, 150, 240, 200])
    for i in range(len(x)):
        cv.circle(image, (x[i], y[i]), 3, (255, 0, 0), -1, 8, 0)
        cv.imwrite("hehe.png", image)
    # 多项式曲线生成
    poly = np.poly1d(np.polyfit(x, y, 6))
    print(poly)
    # 绘制拟合曲线
    for t in range(30, 250, 1):
        y_ = np.int(poly(t))
        cv.circle(image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
    cv.imshow("fit curve", image)
    cv.imwrite("D:/fitcurve.png", image)
    cv.waitKey()
    cv.destroyAllWindows()



# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:18:28 2019
@author: youxinlin
"""

# -----------------------鼠标操作相关------------------------------------------
lsPointsChoose = []
tpPointsChoose = []
pointsCount = 0
count = 0
pointsMax = 6


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, count, pointsMax
    global lsPointsChoose, tpPointsChoose  # 存入选择的点
    global pointsCount  # 对鼠标按下的点计数
    global img2, ROI_bymouse_flag
    img2 = img.copy()  # 此行代码保证每次都重新再原图画  避免画多了
    # -----------------------------------------------------------
    #    count=count+1
    #    print("callback_count",count)
    # --------------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        pointsCount = pointsCount + 1
        # 感觉这里没有用？2018年8月25日20:06:42
        # 为了保存绘制的区域，画的点稍晚清零
        # if (pointsCount == pointsMax + 1):
        #     pointsCount = 0
        #     tpPointsChoose = []
        print('pointsCount:', pointsCount)
        point1 = (x, y)
        print(x, y)
        # 画出点击的点
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)

        # 将选取的点保存到list列表里
        lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
        tpPointsChoose.append((x, y))  # 用于画点
        # ----------------------------------------------------------------------
        # 将鼠标选的点用直线连起来
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        # ----------------------------------------------------------------------
        # ----------点击到pointMax时可以提取去绘图----------------

        cv2.imshow('src', img2)

    # -------------------------右键按下清除轨迹-----------------------------
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        print("right-mouse")
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        cv2.imshow('src', img2)

    # -------------------------双击 结束选取-----------------------------
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # -----------绘制感兴趣区域-----------
        ROI_byMouse()
        ROI_bymouse_flag = 1
        lsPointsChoose = []


def ROI_byMouse():
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array([lsPointsChoose], np.int32)  # pts是多边形的顶点列表（顶点集）
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
    # OpenCV中需要先将多边形的顶点坐标变成顶点数×1×2维的矩阵，再来绘制

    # --------------画多边形---------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    ##-------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    cv2.imshow('mask', mask2)
    cv2.imwrite('mask.jpg', mask2)
    image, contours, hierarchy = cv2.findContours(cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_NONE)
    ROIarea = cv2.contourArea(contours[0])
    print("ROIarea:", ROIarea)
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite('ROI.jpg', ROI)
    cv2.imshow('ROI', ROI)


img = cv2.imread('demo.jpeg')
# ---------------------------------------------------------
# --图像预处理，设置其大小
# height, width = img.shape[:2]
# size = (int(width * 0.3), int(height * 0.3))
# img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
# ------------------------------------------------------------
ROI = img.copy()
cv2.namedWindow('src')
cv2.setMouseCallback('src', on_mouse)
cv2.imshow('src', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




def mergeImg(inputImg, maskImg, contourData, drawPosition):
    '''
    :param inputImg: 输入的图像
    :param maskImg: 输入的模板图像
    :param contourData: 输入的模板中轮廓数据 numpy 形式如[(x1,y1),(x2,y2),...,]
    :param drawPosition: （x,y） 大图中要绘制模板的位置,以maskImg左上角为起始点
    :return: outPutImg：输出融合后的图像
             outContourData: 输出轮廓在inputImg的坐标数据
             outRectData: 输出轮廓的矩形框在inputImg的坐标数据
    '''
    # 通道需要相等
    if inputImg.shape[2] != maskImg.shape[2]:
        print("inputImg shape != maskImg shape")
        return
    inputImg_h = inputImg.shape[0]
    inputImg_w = inputImg.shape[1]
    maskImg_h = maskImg.shape[0]
    maskImg_w = maskImg.shape[1]
    # inputImg图像尺寸不能小于maskImg
    if inputImg_h < maskImg_h or inputImg_w < maskImg_w:
        print("inputImg size < maskImg size")
        return
    # 画图的位置不能超过原始图像
    if ((drawPosition[0] + maskImg_w) > inputImg_w) or ((drawPosition[1] + maskImg_h) > inputImg_h):
        print("drawPosition + maskImg > inputImg range")
        return
    outPutImg = inputImg.copy()
    input_roi = outPutImg[drawPosition[1]:drawPosition[1] + maskImg_h, drawPosition[0]:drawPosition[0] + maskImg_w]
    imgMask_array = np.zeros((maskImg_h, maskImg_w, maskImg.shape[2]), dtype=np.uint8)
    # triangles_list = [np.zeros((len(contourData), 2), int)]
    triangles_list = [contourData]
    cv2.fillPoly(imgMask_array, triangles_list, color=(1, 1, 1))
    cv2.fillPoly(input_roi, triangles_list, color=(0, 0, 0))
    # cv2.imshow('imgMask_array', imgMask_array)
    imgMask_array = imgMask_array * maskImg
    output_ori = input_roi + imgMask_array
    outPutImg[drawPosition[1]:drawPosition[1] + maskImg_h, drawPosition[0]:drawPosition[0] + maskImg_w] = output_ori
    triangles_list[0][:, 0] = contourData[:, 0] + drawPosition[0]
    triangles_list[0][:, 1] = contourData[:, 1] + drawPosition[1]
    outContourData = triangles_list[0]

    return outPutImg, outContourData  # ,outRectData


# imgStr = single_fold_eyelid
# imgMaskStr = single_fold_eyelid
# img = cv2.imread(imgStr)
# maskImg = cv2.imread(imgMaskStr)
# # contourData = np.array([(57, 7), (107, 30), (107, 120), (62, 122), (2, 95), (9, 32)])
# contourData = np.array(list(map(lambda x: (x[1], x[0]), points_choose + points_choose_up[::-1])))
#
# inputImg_h = img.shape[0]
# inputImg_w = img.shape[1]
# maskImg_h = maskImg.shape[0]
# maskImg_w = maskImg.shape[1]
# drawPosition = (15, 0)
# outPutImg = img.copy()
# input_roi = outPutImg
# imgMask_array = np.zeros((maskImg_h, maskImg_w, maskImg.shape[2]), dtype=np.uint8)
# # triangles_list = [np.zeros((len(contourData), 2), int)]
# triangles_list = [contourData]
#
# cv2.fillPoly(imgMask_array, triangles_list, color=(1, 1, 1))
# triangles_list[0][:, 1] = contourData[:, 1] - drawPosition[0]
# cv2.fillPoly(input_roi, triangles_list, color=(0, 0, 0))
# imgMask_array = imgMask_array * maskImg
# cv2.imshow('imgMask_array1', imgMask_array)
# imgMask_array = np.vstack((imgMask_array[drawPosition[0]:], imgMask_array[:drawPosition[0]]))
# cv2.imshow('imgMask_array2', imgMask_array)
# print(input_roi.shape, imgMask_array.shape)
#
# output_ori = input_roi + imgMask_array
# print(output_ori)
# outPutImg = output_ori
# triangles_list[0][:, 0] = contourData[:, 0] + drawPosition[0]
# triangles_list[0][:, 1] = contourData[:, 1] + drawPosition[1]
# outContourData = triangles_list[0]
#
# # outPutImg, outContourData = mergeImg(img, maskImg, contourData, (10, 10))
#
# # print(outPutImg)
# cv2.imshow('2', outPutImg)
# cv2.imshow('3', maskImg)
# cv2.waitKey(0)
