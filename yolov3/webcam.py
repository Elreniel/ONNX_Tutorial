# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:44:05 2021

@author: bcosk
"""

import cv2
import onnx
import onnxruntime
import numpy as np
from PIL import Image
from numpy import asarray


    
def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def returnBoxes(session,img):
    image_data = preprocess(img)
    image_size = np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2)
    
    
    input_name0 = session.get_inputs()[0].name
    input_name1 = session.get_inputs()[1].name
    boxes = session.get_outputs()[0].name
    scores = session.get_outputs()[1].name
    indices = session.get_outputs()[2].name

    boxes,scores,indices = session.run([boxes,scores,indices], {input_name0: image_data, input_name1: image_size})
    global newBoxes, newScores, newLabels
    newBoxes = []
    newScores = []
    newLabels = []

    for temp in indices[:,2]:
        newBoxes.append(boxes[0,temp,:])
        newScores.append(max(scores[0][:,temp]))
        newLabels.append(np.where(scores[0][:,temp] == max(scores[0][:,temp]))[0][0])
    return newBoxes,newScores,newLabels

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    newBoxes = []
    newScores = []
    newLabels = []
    classes = [line.rstrip('\n') for line in open('coco_classes.txt')]
    
    providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider',
    ]
    
    session = onnxruntime.InferenceSession("yolov3-10.onnx",providers=providers)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        img = Image.fromarray(img, 'RGB')
        # detectionThread = threading.Thread(target=returnBoxes, args=(img,))
        # detectionThread.start()
        newBoxes,newScores,newLabels = returnBoxes(session,img)
        img = asarray(img)
        score_threshold = 0.8
        for box, label, score in zip(newBoxes, newLabels, newScores):
            if score > score_threshold:
                color1 = int(label*2)
                color2 = 125 + int(label*2)
                color3 = 255 - int(label*2)
                cv2.rectangle(img, (int(box[1]),int(box[0])), (int(box[3]),int(box[2])),(color1,color2,color3),1)
                cv2.putText(img,classes[label] + ':' + str(np.round(score, 2)), (int(box[1]),int(box[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (color1,color2,color3))
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)
    
if __name__ == '__main__':
    main()
