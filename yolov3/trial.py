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

import numpy as np
from PIL import Image

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data
    
classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.9):
    # Resize boxes
    # ratio = 800.0 / min(image.size[0], image.size[1])
    # boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='b', facecolor='none')
        ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[1], box[0]), color='w', fontsize=12)
        ax.add_patch(rect)
    plt.show()

image = Image.open('photo1.jpg')
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)

session = onnxruntime.InferenceSession("yolov3-10.onnx")
session = onnxruntime.InferenceSession("tiny-yolov3-11.onnx")
input_name0 = session.get_inputs()[0].name
input_name1 = session.get_inputs()[1].name
boxes = session.get_outputs()[0].name
labels = session.get_outputs()[1].name
scores = session.get_outputs()[2].name

result = session.run([boxes,labels,scores], {input_name0: image_data, input_name1: image_size})

boxes = result[0]
scores = result[1]
indices = result[2]

newBoxes = []
newScores = []
newLabels = []

for temp in indices[:,2]:
    newBoxes.append(boxes[0,temp,:])
    newScores.append(max(scores[0][:,temp]))
    newLabels.append(np.where(scores[0][:,temp] == max(scores[0][:,temp]))[0][0])
    
display_objdetect_image(image,newBoxes,newLabels,newScores)