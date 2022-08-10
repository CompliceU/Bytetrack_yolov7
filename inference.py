import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import non_max_suppression, scale_coords
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel

class Detect(object):
    def __init__(self, weights, device, img_size, conf_thres, iou_thres, single_cls=False, half_precision=True, trace= False):
        self.weights = weights 
        self.device =  device 
        self.device = select_device(str(device))
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())

        self.imgsz = img_size 
        self.trace = False
        self.half = half_precision

        self.conf_thres = conf_thres 
        self.iou_thres = iou_thres 
    
        if trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once for warmup
     
    def detecte(self, im0):
        img = letterbox(im0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False) #for filtering change classes  
            dets = []
            for i, det in enumerate(pred):  # detections per image     
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):     
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if self.names[int(cls)] in ['car', 'person', 'truck'] and conf > 0.5:
                            dets.append([xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item(), conf.item()])
            return dets


    

