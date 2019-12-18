#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       pengj
    @   date    :       2019/12/2 10:37
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       https://sm.ms/doc/v2#277af2f2a6a9c6679bfc62a51b714c0d
    Python requests库处理 multipart/form-data 请求以及 boundary值问题,看下面
    https://blog.csdn.net/Enderman_xiaohei/article/details/89421773
-------------------------------------------------
"""
import json

__author__ = 'Max_Pengjb'

import requests

username = 'Max_pengjb'
password = 'sd811811'
api_url = 'https://sm.ms/api/v2'
login_url = api_url + '/token'
upload_url = api_url + '/upload'
upload_headers = {'Authorization': None}

payload = {'username': username, 'password': password}
r = requests.post(login_url, params=payload)
res = r.json()
if 'success' in res and res['success'] == 'True':
    token = res['data']['token']
print(token)
upload_headers['Authorization'] = token
files = {'smfile': open('hehe.png', 'rb')}
r = requests.post(upload_url, headers=upload_headers, files=files)
json_content = json.loads(r.text)
print("1: ", r.text)
print("2: ", r.request.body)
print("3: ", r.request.headers)
print(json_content)
if 'success' in res and res['success'] == 'True':
    print(json_content['data'])
    print(json_content['data']['url'])
else:
    print(json_content['message'])
