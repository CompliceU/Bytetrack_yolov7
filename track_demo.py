import argparse
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, TracedModel
import time 
from tracker.byte_tracker import BYTETracker
from utils.visualize import plot_tracking
from tracking_utils.timer import Timer
from inference import Detect

def track_demo():
    video_path = "video/palace.mp4" 
    txt_dir = "result"
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    # Detected
    conf_thres = 0.25
    iou_thres = 0.25
    img_size = 640
    weights = "weights/yolov7-tiny.pt"
    device = 0
    half_precision = True
    deteted = Detect(weights, device, img_size, conf_thres, iou_thres, single_cls=False, half_precision=half_precision, trace= False)
   
    # Tracking
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    frame_rate = 25
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20_check = False
    res_file = os.path.join(txt_dir, video_path.split('/')[-1].replace('mp4', 'txt'))

    print(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    timer = Timer()
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    results = []
    while True:
        _,im0 = cap.read()
        frame_id += 1
        if _:
            height, width, _ = im0.shape
            t1 = time.time()
            dets = deteted.detecte(im0)
            online_targets = tracker.update(np.array(dets), [height, width], (height, width))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save result for evaluation
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            t2 = time.time()
            print(f"FPS:{1 /(t2-t1):.2f}")
            timer.toc()
            #print(1. / timer.average_time)
            online_im = plot_tracking(im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / 1 /(t2-t1))
            with open(res_file, 'w') as f:
                f.writelines(results)
            cv2.imshow("Frame", online_im)
            ch = cv2.waitKey(1)  # 1 millisecond
            if ch == ord("q") : 
                break
        else:
            break
if __name__ == "__main__":
    track_demo()