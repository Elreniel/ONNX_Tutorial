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
from random import randrange

def returnBoxes(session,img):
    image_data = preprocess(img)
    input_name = session.get_inputs()[0].name
    boxes = session.get_outputs()[0].name
    labels = session.get_outputs()[1].name
    scores = session.get_outputs()[2].name
    masks = session.get_outputs()[3].name

    return session.run([boxes,labels,scores,masks], {input_name: image_data})

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

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    newBoxes = []
    newScores = []
    newLabels = []
    newMasks = []
    classes = [line.rstrip('\n') for line in open('coco_classes.txt')]
    
    providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider',
    ]
    
    session = onnxruntime.InferenceSession("MaskRCNN-10.onnx",providers=providers)
    
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        img = Image.fromarray(img, 'RGB')
        newBoxes,newLabels,newScores,newMasks = returnBoxes(session,img)
        
        ratio = 800.0 / min(img.size[0], img.size[1])
        newBoxes /= ratio
        img = asarray(img)
        
        score_threshold = 0.8
        for box, label, score,mask in zip(newBoxes,newLabels,newScores,newMasks):
            if score > score_threshold:
                
                mask = mask[0, :, :, None]
                int_box = [int(i) for i in box]
                mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
                mask = mask > 0.5
                im_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                x_0 = max(int_box[0], 0)
                x_1 = min(int_box[2] + 1, img.shape[1])
                y_0 = max(int_box[1], 0)
                y_1 = min(int_box[3] + 1, img.shape[0])
                mask_y_0 = max(y_0 - box[1], 0)
                mask_y_1 = mask_y_0 + y_1 - y_0
                mask_x_0 = max(x_0 - box[0], 0)
                mask_x_1 = mask_x_0 + x_1 - x_0
                im_mask[y_0:y_1, x_0:x_1] = mask[
                    mask_y_0 : mask_y_1, mask_x_0 : mask_x_1
                ]
                im_mask = im_mask[:, :, None]

                # OpenCV version 4.x
                contours, hierarchy = cv2.findContours(
                    im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                img = cv2.drawContours(img, contours, -1, 25, 3)
                
                color1 = int(label*2)
                color2 = 125 + int(label*2)
                color3 = 255 - int(label*2)
                cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])),(color1,color2,color3),1)
                cv2.putText(img,classes[label] + ':' + str(np.round(score, 2)), (int(box[1]),int(box[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (color1,color2,color3))
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
        
def main():
    show_webcam(mirror=True)
    
if __name__ == '__main__':
    main()


