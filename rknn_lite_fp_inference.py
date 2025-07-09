#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN Lite 2 测试rknn模型对单张图片进行预处理和推理。
python3 rknn_lite_fp_inference.py --model best_fp.rknn --image test.jpg --names "class_A" "class_B"
"""

import os
import cv2
import numpy as np
import argparse
from rknnlite.api import RKNNLite

def preprocess(image_path, input_size=640):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    original_img = img.copy()
    h, w = img.shape[:2]
    
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_img = cv2.resize(img, (new_w, new_h))
    padded_img = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded_img[:new_h, :new_w] = resized_img
    
    return original_img, padded_img, scale

def postprocess_fp(rknn_outputs, scale, conf_threshold=0.25, iou_threshold=0.45, class_names=None):
    float_output = rknn_outputs[0][0].astype(np.float32)
        
    if float_output.shape[0] < float_output.shape[1]:
        float_output = float_output.T

    boxes, scores, class_ids = [], [], []
    for det in float_output:
        cx, cy, w, h = det[:4]
        class_conf = det[4:]
        class_id = np.argmax(class_conf)
        conf = class_conf[class_id]
        
        if conf >= conf_threshold:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(class_id)
    
    if not boxes:
        return []
        
    indices = cv2.dnn.NMSBoxes(boxes, np.array(scores), conf_threshold, iou_threshold)
    
    final_detections = []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        final_detections.append({
            'bbox': [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)],
            'confidence': float(scores[i]),
            'class_id': int(class_ids[i]),
            'class_name': class_names[class_ids[i]] if class_names else f"class_{class_ids[i]}"
        })
    return final_detections

def draw_results(image, detections, save_path="result_fp.jpg"):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        conf = det['confidence']
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    cv2.imwrite(save_path, image)
    print(f"✓ 检测结果图已保存至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RKNN Lite 2 FP Model Inference Script")
    parser.add_argument('--model', type=str, required=True, help='Path to the FP .rknn model file')
    parser.add_argument('--image', type=str, required=True, help='Path to the test image')
    parser.add_argument('--names', nargs='+', required=True, help="Class names, separated by space")
    
    args = parser.parse_args()

    rknn_lite = RKNNLite(verbose=False)
    
    print(f">>> Loading RKNN model: {args.model}")
    ret = rknn_lite.load_rknn(args.model)
    if ret != 0:
        print("Error: Failed to load RKNN model.")
        exit(ret)
    
    print(">>> Initializing runtime...")
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Error: Failed to initialize runtime environment.")
        exit(ret)
    print("✓ Runtime initialized successfully.")
    
    original_img, padded_img, scale = preprocess(args.image)
    
    print(">>> Running inference...")

    input_data = np.expand_dims(padded_img, axis=0)

    outputs = rknn_lite.inference(inputs=[input_data])

    
    detections = postprocess_fp(outputs, scale, class_names=args.names)
    
    print("\n--- Detection Results ---")
    if detections:
        for i, det in enumerate(detections):
            print(f"  {i+1}. Class: {det['class_name']}, Confidence: {det['confidence']:.3f}, BBox: {det['bbox']}")
        draw_results(original_img, detections)
    else:
        print("  No objects detected.")
        
    rknn_lite.release()
