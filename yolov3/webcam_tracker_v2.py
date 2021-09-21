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
import multiprocessing
from multiprocessing import Process

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

def returnBoxes(session,img,newBoxes,newScores,newLabels):
    
    image_data = preprocess(img)
    image_size = np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2)
            
    input_name0 = session.get_inputs()[0].name
    input_name1 = session.get_inputs()[1].name
    boxes = session.get_outputs()[0].name
    scores = session.get_outputs()[1].name
    indices = session.get_outputs()[2].name

    boxes,scores,indices = session.run([boxes,scores,indices], {input_name0: image_data, input_name1: image_size})
    newBoxes[:] = []
    newScores[:] = []
    newLabels[:] = []

    if selectedModel == 0:
        for temp in indices[:,2]:
            tempBoxes = tuple([int(boxes[0,temp,:][1]),int(boxes[0,temp,:][0]),int(boxes[0,temp,:][3]-boxes[0,temp,:][1]),int(boxes[0,temp,:][2]-boxes[0,temp,:][0])])
            newBoxes.append(tempBoxes)
            newScores.append(max(scores[0][:,temp]))
            newLabels.append(np.where(scores[0][:,temp] == max(scores[0][:,temp]))[0][0])
    elif selectedModel == 1:
        for temp in indices[:,:,2][0]:
            tempBoxes = tuple([int(boxes[0,temp,:][1]),int(boxes[0,temp,:][0]),int(boxes[0,temp,:][3]-boxes[0,temp,:][1]),int(boxes[0,temp,:][2]-boxes[0,temp,:][0])])
            newBoxes.append(tempBoxes)
            newScores.append(max(scores[0][:,temp]))
            newLabels.append(np.where(scores[0][:,temp] == max(scores[0][:,temp]))[0][0])

def initiateTrackers(img,newBoxes):
    img = asarray(img)
    
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[5]
    
    trackerList = []
    for box in newBoxes:       
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            
        tracker.init(img, box)
        trackerList.append(tracker)
    
    return trackerList

if __name__ == '__main__':
    
    mirror = True
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    classes = [line.rstrip('\n') for line in open('coco_classes.txt')]
     
    global manager, newBoxes, newLabels,newScores
    manager = multiprocessing.Manager()
    newBoxes = manager.list()
    newLabels = manager.list()
    newScores = manager.list()
        
    ret_val, img = cam.read()
    if mirror: 
        img = cv2.flip(img, 1)
    img = Image.fromarray(img, 'RGB')
    
    selectedModel = 0 # 0 for YOLO, 1 for Tiny-YOLO
    
    providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider',
    ]

    if selectedModel == 0:
        session = onnxruntime.InferenceSession("yolov3-10.onnx", providers=providers)
    elif selectedModel == 1:
        session = onnxruntime.InferenceSession("tiny-yolov3-11.onnx", providers=providers)
    
    p = Process(target=returnBoxes,args=[session,img,newBoxes,newScores,newLabels])
    p.start()
    
    trackerList = initiateTrackers(img,newBoxes)
        
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        img = Image.fromarray(img, 'RGB')
        
        if not p.is_alive():
            p = Process(target=returnBoxes,args=[session,img,newBoxes,newScores,newLabels])
            p.start()                        
            trackerList = initiateTrackers(img, newBoxes)
        
        img = asarray(img)
        score_threshold = 0.8
        
        for tracker, box, label, score in zip(trackerList, newBoxes, newLabels, newScores):
            if score > score_threshold:
                ok, box = tracker.update(img)
                color1 = int(label*2)
                color2 = 125 + int(label*2)
                color3 = 255 - int(label*2)
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(img, p1, p2,(color1,color2,color3),1)
                cv2.putText(img,classes[label] + ':' + str(np.round(score, 2)), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (color1,color2,color3))
        
        cv2.imshow('Object Tracker', img)
        if cv2.waitKey(1) == 27: 
            break
            
    try:
        p.kill()
    except:
        pass      
    
    cam.release()
    cv2.destroyAllWindows()


