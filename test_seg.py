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
# model = YOLO('runs/segment/train31/weights/best.pt') #已经训练好的模型
mode_path = "runs/detect/yolov8n_add_head2/weights/best.pt"
# mode_path = "runs/detect/train8/weights/last.pt"
model = YOLO(mode_path) #已经训练好的模型

# Define path to the image file
# source = 'ultralytics/assets/bus.jpg' #待预测的数据保存路径
source = '/work/datasets/coco128/images/train2017' #待预测的数据保存路径
# Run inference on the source
results = model(source, mode='predict', save=True)  # list of Results objects



## export model onnx
# 指定输入和输出导出onnx
# input_names = [ "input_1"]
# output_names = [ "output1" ]
# torch.onnx.export(model, (dummy_input1, dummy_input2), "name.onnx", verbose=True, input_names=input_names, output_names=output_names)

print("===========  onnx =========== ")
model.export(format='onnx',opset=12,simplify=True)
print("======================== convert onnx Finished! .... ")
# from onnxsim import simplify
# import onnx
# onnx_model = onnx.load(mode_path.replace("pt","onnx"))  # load onnx model
# model_simp, check = simplify(onnx_model)
# assert check, "Simplified ONNX model could not be validated"
# onnx.save(model_simp, mode_path.replace("pt","_simplify.onnx"))
# print('finished exporting onnx')
