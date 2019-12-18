# 导入系统库并定义辅助函数
from pprint import pformat

# import PythonSDK
from PythonSDK.facepp import API, File

# 导入图片处理类
import PythonSDK.ImagePro

# 以下四项是dmeo中用到的图片资源，可根据需要替换
detech_img_url = 'http://bj-mc-prod-asset.oss-cn-beijing.aliyuncs.com/mc-official/images/face/demo-pic11.jpg'
faceSet_img = './imgResource/demo.jpeg'  # 用于创建faceSet
face_search_img = './imgResource/search.png'  # 用于人脸搜索
segment_img = './imgResource/segment.jpg'  # 用于人体抠像
merge_img = './imgResource/merge.jpg'  # 用于人脸融合
single_fold_eyelid = './imgResource/single_fold_eyelid5.jpg'  # 单眼皮图片
double_fold_eyelid = './imgResource/double_fold_eyelid5.jpg'  # face++人脸融合后的双眼皮照片
double_fold_merge = './imgResource/double_fold_merge.jpg'  # face++ 双眼皮照片


# 此方法专用来打印api返回的信息
def print_result(hit, result):
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))


def printFuctionTitle(title):
    return "\n" + "-" * 60 + title + "-" * 60;


# 初始化对象，进行api的调用工作
api = API()
# -----------------------------------------------------------人脸识别部分-------------------------------------------
# 人脸检测：https://console.faceplusplus.com.cn/documents/4888373
# res = api.detect(image_url=detech_img_url, return_attributes="gender,age,smiling,headpose,facequality,"
#                                                              "blur,eyestatus,emotion,ethnicity,beauty,"
#                                                              "mouthstatus,skinstatus,eyegaze")
# 人脸分析 https://console.faceplusplus.com.cn/documents/4888383
# res = api.face.analyze(face_tokens="f914dfbca818c802acc07840dae19214",
#                        return_landmark=2, image_url=detech_img_url,
#                        return_attributes="gender,age,smiling,headpose,facequality,"
#                                          "blur,eyestatus,emotion,ethnicity,beauty,"
#                                          "mouthstatus,skinstatus,eyegaze"
#                        )

# 皮肤分析API https://console.faceplusplus.com.cn/documents/119745378
# res = api.skinanalyze(image_url=detech_img_url)

# 面部特征分析API https://console.faceplusplus.com.cn/documents/118131136
# res = api.facialfeatures(image_file=File(single_fold_eyelid), return_imagereset=1)

# print_result(printFuctionTitle("人脸检测"), res)

# 把face++返回的base64的图片信息解码，存为图片
import os, base64
# imgdata = base64.b64decode(res.image_reset)
# with open("./haha.png", "wb") as f2:
#     f2.write(imgdata)

import cv2

print(cv2.__version__)
img = cv2.imread(single_fold_eyelid)
img2 = cv2.imread(double_fold_eyelid)


def get_points_from_facepp(img):
    # 下面是描点，把特征点全部用红色的像素画出来，描点开始
    point_color = (0, 0, 255)  # BGR
    thickness = 0  # 可以为 0 、4、8 正值表示圆边框宽度. 负值表示画一个填充圆形
    points_list = res.denselandmark
    left_eye_pupil_center = {}  # 左眼瞳孔中心位置
    left_eye_pupil_radius = 0  # 左眼瞳孔半径
    right_eye_pupil_center = {}
    right_eye_pupil_radius = 0

    left_points_choose = []
    right_points_choose = []
    for i in range(32):
        left_eyelid = points_list.left_eye_eyelid["left_eye_eyelid_" + str(i)]
        right_eyelid = points_list.right_eye_eyelid["right_eye_eyelid_" + str(i)]
        left_points_choose.append([left_eyelid.y, left_eyelid.x])
        right_points_choose.append([right_eyelid.y, right_eyelid.x])
        # points_choose_up.append([left_eyelid.y - 20, left_eyelid.x])
        # points_choose_up_2.append([left_eyelid.y - 35, left_eyelid.x])
        img[left_eyelid.y, left_eyelid.x] = point_color
        img[right_eyelid.y, right_eyelid.x] = point_color
        # img[left_eyelid.y - 10, left_eyelid.x] = (255, 255, 0)

    # 画出所有的特征点
    for k, points in points_list.items():
        print(k, "->", points)
        for name, point in points.items():
            # print(name, point)
            if name == "left_eye_pupil_center":
                left_eye_pupil_center = point
            elif name == "left_eye_pupil_radius":
                left_eye_pupil_radius = point
            elif name == "right_eye_pupil_center":
                right_eye_pupil_center = point
            elif name == "right_eye_pupil_radius":
                right_eye_pupil_radius = point
            else:
                pass
                # img[point.y, point.x] = point_color
                # cv2.circle(img, (point.x, point.y), point_size, point_color, thickness)
    cv2.circle(img, (left_eye_pupil_center.x, left_eye_pupil_center.y), left_eye_pupil_radius, point_color, thickness)
    cv2.circle(img, (right_eye_pupil_center.x, right_eye_pupil_center.y), right_eye_pupil_radius, point_color,
               thickness)
    # 上面是描点，把特征点全部用红色的像素画出来，描点结束
    return left_points_choose + right_points_choose


imgInfo = img.shape
print(imgInfo)  # (2576, 1932, 3) 图片信息， 依次是  高 宽 一个像素点由3个像素组成
# cv2.imshow('before', img)
# points1 = get_points_from_facepp(img)
# points2 = get_points_from_facepp(img2)
# print(points1 == points2)
# cv2.imshow('after', img)
# cv2.imshow('after', img2)
# cv2.imwrite('img2_after.png', img2)  # 保存图片
# 等待按键输入
# cv2.waitKey()
# 关闭所有窗口，释放资源
# cv2.destroyAllWindows()

# from matplotlib import pyplot as plt
# import numpy as np
# print(img.shape)
# plt.imshow(img)
# plt.show()

# 人脸比对：https://console.faceplusplus.com.cn/documents/4887586
# compare_res = api.compare(image_file1=File(face_search_img), image_file2=File(face_search_img))
# print_result("compare", compare_res)

# 人脸搜索：https://console.faceplusplus.com.cn/documents/4888381
# 人脸搜索步骤
# 1,创建faceSet:用于存储人脸信息(face_token)
# 2,向faceSet中添加人脸信息(face_token)
# 3，开始搜索

# 删除无用的人脸库，这里删除了，如果在项目中请注意是否要删除
# api.faceset.delete(outer_id='faceplusplus', check_empty=0)
# # 1.创建一个faceSet
# ret = api.faceset.create(outer_id='faceplusplus')
#
# # 2.向faceSet中添加人脸信息(face_token)
# faceResStr=""
# res = api.detect(image_file=File(faceSet_img))
# faceList = res["faces"]
# for index in range(len(faceList)):
#     if(index==0):
#         faceResStr = faceResStr + faceList[index]["face_token"]
#     else:
#         faceResStr = faceResStr + ","+faceList[index]["face_token"]
#
# api.faceset.addface(outer_id='faceplusplus', face_tokens=faceResStr)
#
# # 3.开始搜索相似脸人脸信息
# search_result = api.search(image_file=File(face_search_img), outer_id='faceplusplus')
# print_result('search', search_result)

# -----------------------------------------------------------人体识别部分-------------------------------------------

# 人体抠像:https://console.faceplusplus.com.cn/documents/10071567
# segment_res = api.segment(image_file=File(segment_img))
# f = open('./imgResource/demo-segment.b64', 'w')
# f.write(segment_res["result"])
# f.close()
# print_result("segment", segment_res)
# # 开始抠像
# PythonSDK.ImagePro.ImageProCls.getSegmentImg("./imgResource/demo-segment.b64")

# -----------------------------------------------------------证件识别部分-------------------------------------------
# 身份证识别:https://console.faceplusplus.com.cn/documents/5671702
# ocrIDCard_res = api.ocridcard(image_url="https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/"
#                                         "c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=7a16a1be19178a82da3177f2976a18e8"
#                                         "/902397dda144ad34a1b2dcf5d7a20cf431ad85b7.jpg")
# print_result('ocrIDCard', ocrIDCard_res)

# 银行卡识别:https://console.faceplusplus.com.cn/documents/10069553
# ocrBankCard_res = api.ocrbankcard(image_url="http://pic.5tu.cn/uploads/allimg/1107/191634534200.jpg")
# print_result('ocrBankCard', ocrBankCard_res)

# -----------------------------------------------------------图像识别部分-------------------------------------------
# 人脸融合：https://console.faceplusplus.com.cn/documents/20813963
# template_rectangle参数中的数据要通过人脸检测api来获取

# template_url list:
# https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574980389347&di=a98287b30386b159d3be8067d5b8df00&imgtype=0&src=http%3A%2F%2Fgss0.baidu.com%2F7LsWdDW5_xN3otqbppnN2DJv%2Fzhidao%2Fpic%2Fitem%2F5d6034a85edf8db15af248710523dd54574e74c2.jpg
# http://i1.hdslb.com/bfs/archive/f391e3d633ca7fb2bcc1ac68cea380ec8b2c87f5.jpg
# https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981285576&di=1cf524ef3fb08beea100ac79d917e752&imgtype=0&src=http%3A%2F%2Fimg.mp.itc.cn%2Fupload%2F20170501%2F4274ec793fcf4b6098a67750892d900a_th.jpeg
# http://i2.hdslb.com/bfs/archive/48acf9efd46e9319c4dc05c5c60cdcc43d5985a7.jpg
# https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981356999&di=1d4bd18dbce31f44604031b157fd26ef&imgtype=0&src=http%3A%2F%2Fwx3.sinaimg.cn%2Flarge%2F006byqyGly1fbp1ymowaqj30ji0shn0c.jpg
# https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=204522960,1226759988&fm=15&gp=0.jpg
# https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981404543&di=dc51eb45956cd1ff86b2efbc5bfeee74&imgtype=0&src=http%3A%2F%2Fe0.ifengimg.com%2F03%2F2019%2F0414%2F5E4A8223AEDC57597EEC455E57EC6ECBBFE63678_size81_w750_h938.jpeg
# https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981508096&di=ab20baafd546e408b84b02da1c769a4a&imgtype=0&src=http%3A%2F%2Fwww.jder.net%2Fwp-content%2Fuploads%2F2017%2F11%2F20171108130848-90.jpg
# https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981557927&di=484eae05e3ed0c0f4d30914862a012a0&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20181110%2F2063daae7ad94d3294d21fda4d604a6b.jpeg
# https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=76125465,4026057848&fm=26&gp=0.jpg
mergeFace_res = api.mergeface(
    template_url='https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1574981557927&di=484eae05e3ed0c0f4d30914862a012a0&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20181110%2F2063daae7ad94d3294d21fda4d604a6b.jpeg',
    merge_url='https://cdn.faceplusplus.com.cn/mc-official/scripts/demoScript/images/demo-pic114.jpg')
# mergeFace_res = api.mergeface(template_base64=template_base64, merge_base64=merge_base64)
# print_result("mergeFace", mergeFace_res)
#
# # 开始融合
print(type(mergeFace_res))
imgdata_str = base64.b64decode(mergeFace_res["result"])
import numpy as np
print(mergeFace_res)
nparr = np.fromstring(imgdata_str, np.uint8)
print(nparr)
imgdata = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cv2.imshow("hebing", imgdata)
# 等待按键输入
cv2.waitKey()
# 关闭所有窗口，释放资源
cv2.destroyAllWindows()
# PythonSDK.ImagePro.ImageProCls.getMergeImg(mergeFace_res["result"])

"""
函数详解：
addWeighted(InputArray_src1, 
            double_alpha, 
            InputArray_src2, 
            double_beta, 
            double_gamma, 
            OutputArray_dst, 
            int_dtype=-1
            );
一共有七个参数：前4个是两张要合成的图片及它们所占比例，
                            第5个double gamma起微调作用，
                            第6个OutputArray dst是合成后的图片，
                            第7个输出的图片的类型（可选参数，默认-1）
"""
