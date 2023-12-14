#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/18 9:33
# @Author  : rzh
# @Site    : 
# @File    : train_seg.py
# @Software: PyCharm

import sys
import os
print('Python %s on %s' % (sys.version, sys.platform))

print(os.getcwd ())

# train model
from ultralytics import YOLO
# Load a model 三选一
# model = YOLO('./ultralytics/cfg/models/v8/yolov8-seg.yaml')  # build a new model from YAML 配置好模型和训练数据yaml文件信息
# model = YOLO('ultralytics/cfg/models/v8/yolov8_add_head.yaml')  # load a pretrained model (recommended for training)

# model = YOLO('yolov8n.yaml')  # build from YAML and transfer weights
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model = YOLO('ultralytics/cfg/models/v8/yolov8_add_head.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# model = YOLO('yolov8n-seg.pt')



# Train the model
# mode_path = "runs/detect/train8/weights/last.pt"
# results = model.train(data='./ultralytics/cfg/datasets/coco128.yaml', epochs=300, imgsz=640,resume=True)
model.train(data='./ultralytics/cfg/datasets/coco128.yaml',batch=16,epochs=300, imgsz=640,name='yolov8n_add_head')

# model('https://ultralytics.com/images/bus.jpg')
model('ultralytics/assets/bus.jpg')

