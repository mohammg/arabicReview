# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 14:41:53 2021

@author: moham
"""
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'text':'للاسف الكتاب مش فى المستوى وخصوصا المقالات اللى ف جزء التوابل'})

print(r.json())