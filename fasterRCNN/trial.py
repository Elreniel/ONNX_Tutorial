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

def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

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
img_data = preprocess(img)

session = onnxruntime.InferenceSession("faster_rcnn_R_50_FPN_1x.onnx")
input_name = session.get_inputs()[0].name
boxes = session.get_outputs()[0].name
labels = session.get_outputs()[1].name
scores = session.get_outputs()[2].name

result = session.run([boxes,labels,scores], {input_name: img_data})

display_objdetect_image(img, result[0], result[1], result[2])