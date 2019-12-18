#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       pengj
    @   date    :       2019/11/29 1:03
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""
import time
from typing import List

from flask import json

__author__ = 'Max_Pengjb'
start_time = time.time()

# 下面写上代码块
# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time

http_url = 'https://api-cn.faceplusplus.com/imagepp/v1/mergeface'
key = "30Wy6XzWFPpPT4s76F-KVWgSHBDi22sF"
secret = "OpSI-R2SOQTo9Re6v28XdtmNBPKiUt5V"
filepath = r'./imgResource/segment.jpg'  # 用于人体抠像

segment_img = './imgResource/single_fold_eyelid5.jpg'  # 用于人体抠像
merge_img = './imgResource/double_fold_merge.jpg'  # 用于人脸融合

boundary = '----------%s' % hex(int(time.time() * 1000))
data = []

data.append('--%s' % boundary)
data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
data.append(key)

data.append('--%s' % boundary)
data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
data.append(secret)

data.append('--%s' % boundary)
fr = open(filepath, 'rb')
data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'template_file')
data.append('Content-Type: %s\r\n' % 'application/octet-stream')
data.append(fr.read())
fr.close()

data.append('--%s' % boundary)
fr = open(merge_img, 'rb')
data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'merge_file')
data.append('Content-Type: %s\r\n' % 'application/octet-stream')
data.append(fr.read())
fr.close()

data.append('--%s--\r\n' % boundary)

for i, d in enumerate(data):
    if isinstance(d, str):
        data[i] = d.encode('utf-8')

http_body = b'\r\n'.join(data)

# build http request
req = urllib.request.Request(url=http_url, data=http_body)

# header
req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

try:
    # post data to server
    resp = urllib.request.urlopen(req, timeout=5)
    # get response
    qrcont = resp.read()
    # if you want to load as json, you should decode first,
    # for example: json.loads(qrcount.decode('utf-8'))
    print(qrcont.decode('utf-8'))
    mergeFace_res = json.loads(qrcont)
    import PythonSDK.ImagePro
    print(mergeFace_res["result"])
    PythonSDK.ImagePro.ImageProCls.getMergeImg(mergeFace_res["result"])
except urllib.error.HTTPError as e:
    print(e.read().decode('utf-8'))

# 上面中间写上代码块
end_time = time.time()
print('Running time: %s Seconds' % (end_time - start_time))



