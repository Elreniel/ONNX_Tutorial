# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:53:11 2021

@author: bcosk
"""

import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper

def img_preprocessing(path):
    img = cv2.imread(path)
    img = np.dot(img[...,:3],[0.299,0.587,0.114])
    img = cv2.resize(img,dsize=(28,28),interpolation = cv2.INTER_AREA)
    img.resize((1,1,28,28))
    return img

img3 = img_preprocessing("three.png")
img4 = img_preprocessing("four.jpg")
img8 = img_preprocessing("eight.jpg")
img6 = img_preprocessing("six.png")
img7 = img_preprocessing("seven.png")

data = json.dumps({'data': img7.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')

session = onnxruntime.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: data})
prediction = int(np.argmax(np.array(result).squeeze(),axis=0))
print(prediction)