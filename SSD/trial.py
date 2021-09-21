# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:10:49 2021

@author: bcosk
"""

import numpy as np
from PIL import Image
import onnx
import onnxruntime
from onnx import numpy_helper
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess(img_path):
    input_shape = (1, 3, 1200, 1200)
    img = Image.open(img_path)
    img = img.resize((1200, 1200), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.9):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)
    plt.show()

img = Image.open('photo1.jpg')
img_data = preprocess('photo1.jpg')

session = onnxruntime.InferenceSession("ssd-10.onnx")
input_name = session.get_inputs()[0].name
boxes = session.get_outputs()[0].name
labels = session.get_outputs()[1].name
scores = session.get_outputs()[2].name

result = session.run([boxes,labels,scores], {input_name: img_data})

display_objdetect_image(img, result[0][0], result[1][0], result[2][0])
