#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/20 12:40
# @Author  : rzh
# @Site    : 
# @File    : test_seg.py
# @Software: PyCharm

#predict
from ultralytics import YOLO
import cv2
import os,datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'    #如果训练时候报Initializing libiomp5md.dll, but found libiomp5md.dll already initialized错误，加上这句
model = YOLO('runs/segment/train31/weights/best.pt') #已经训练好的模型
# Define path to the image file
source = '/F_Pan/pro_ai/yoloV8-/datasets/split/images/test' #待预测的数据保存路径
# Run inference on the source
results = model(source, mode='predict', save=True)  # list of Results objects