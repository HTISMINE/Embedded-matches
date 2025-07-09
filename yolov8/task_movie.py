
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from ultralytics import YOLO
import cv2

# 初始化模型
model1 = YOLO('/yolov8s.pt')
model2 = YOLO('/best.pt')

# 视频处理
cap = cv2.VideoCapture('.mp4')  # 支持RTSP/摄像头输入
out = cv2.VideoWriter('output.mp4', 
                     cv2.VideoWriter_fourcc(*'mp4v'),
                     30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 执行第一个模型的推理
    results1 = model1(frame,
                   classes=0,
                   device='cuda:3')  # GPU加速
    
    # 绘制第一个模型的检测结果
    annotated_frame1 = results1[0].plot()
    
    # 执行第二个模型的推理，使用第一个模型的输出作为输入
    results2 = model2(annotated_frame1, 
                   device='cuda:3')  # GPU加速
    
    # 绘制第二个模型的检测结果
    annotated_frame2 = results2[0].plot()
    
    # 输出处理
    out.write(annotated_frame2)

cap.release()
out.release()