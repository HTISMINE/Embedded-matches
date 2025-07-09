import cv2
import numpy as np
import argparse
import time
import os
import threading
import queue
import signal
import sys
import subprocess
from rknnlite.api import RKNNLite

INPUT_SIZE = 640
CUSTOM_CLASS_NAMES = ["hatch_COVER_CLOSE", "hatch_COVER_OPEN"]
COCO_CLASS_NAMES = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class VideoCaptureThread:
    def __init__(self, url, buffer_size=2):
        self.url, self.frame_queue, self.running = url, queue.Queue(maxsize=buffer_size), False
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
    
    def _capture_loop(self):
        while self.running:
            print(f"INFO: [CaptureThread] 正在尝试连接: {self.url}")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"错误: [CaptureThread] 无法打开视频流. 5秒后重试...")
                time.sleep(5)
                continue
            print("INFO: [CaptureThread] 视频流连接成功.")
            while self.running:
                ret, frame = cap.read()
                if not ret: break
                if not self.frame_queue.full():
                    try: self.frame_queue.put(frame, timeout=0.1)
                    except queue.Full: pass
            cap.release()
    
    def start(self): 
        self.running = True
        self.capture_thread.start()
    
    def get_frame_queue(self): 
        return self.frame_queue
    
    def stop(self): 
        self.running = False
        self.capture_thread.join(timeout=2)

class AudioManager:
    def __init__(self, audio_file):
        self.audio_file, self.process, self.last_play_time, self.min_interval = audio_file, None, 0, 5.0
    
    def play(self):
        if not self.audio_file or not os.path.exists(self.audio_file): return
        is_playing = self.process is not None and self.process.poll() is None
        if not is_playing and (time.time() - self.last_play_time > self.min_interval):
            print("警报: 检测到 'OPEN' 状态, 播放声音...")
            try:
                self.process = subprocess.Popen(['aplay', '-D', 'hw:1,0', self.audio_file], 
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.last_play_time = time.time()
            except Exception as e: 
                print(f"错误: 播放音频失败: {e}")
    
    def cleanup(self):
        if self.process and self.process.poll() is None: 
            self.process.kill()

class StreamProcessor:
    def __init__(self, args):
        self.args = args
        self.frame_queue, self.result_queue = None, queue.Queue(maxsize=2)
        self.running = True
        self.fps, self.frame_count, self.start_time = 0, 0, time.time()
        self._init_models()
    
    def _init_models(self):
        print("--- 加载自定义模型 ---")
        self.custom_rknn, self.custom_qnt_info = self._load_single_model(self.args.custom_model, self.args.custom_model_type)
        print("\n--- 加载官方模型 ---")
        self.official_rknn, self.official_qnt_info = self._load_single_model(self.args.official_model, self.args.official_model_type)
    
    def _load_single_model(self, model_path, model_type):
        rknn_lite = RKNNLite(verbose=False)
        
        # Load the model
        if rknn_lite.load_rknn(model_path) != 0: 
            raise RuntimeError(f"加载模型失败: {model_path}")
        
        # Initialize runtime
        if rknn_lite.init_runtime() != 0: 
            raise RuntimeError("初始化运行环境失败")
        
        qnt_info = None
        
        # For quantized models, we need to handle dequantization
        if model_type in ['int8', 'i16']:
            # Default quantization parameters for YOLOv5/v8 models
            if model_type == 'int8':
                qnt_info = [{
                    'qnt_type': 'affine',
                    'zp': 0,  # Zero point for symmetric quantization
                    'scale': 0.00390625  # 1/256 for int8
                }]
            else:  # i16
                qnt_info = [{
                    'qnt_type': 'affine',
                    'zp': 0,
                    'scale': 0.0000305176  # 1/32768 for int16
                }]
            
            print(f"✓ 模型 {os.path.basename(model_path)} 初始化成功 (类型: {model_type.upper()}, 需要反量化)")
        else:
            print(f"✓ 模型 {os.path.basename(model_path)} 初始化成功 (类型: {model_type.upper()}, 无需反量化)")
        
        
        sdk_version = rknn_lite.get_sdk_version()
        print(f"  SDK版本: {sdk_version}")
        
        return rknn_lite, qnt_info
    
    def _inference_loop(self):
        while self.running:
            try: 
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty: 
                continue
            
            h, w = frame.shape[:2]
            scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            
            padded_img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
            padded_img[:new_h, :new_w] = resized
            input_data = np.expand_dims(padded_img, axis=0)
            
            # Run inference on custom model
            custom_outputs = self.custom_rknn.inference(inputs=[input_data])
            custom_detections = postprocess(custom_outputs, scale, self.custom_qnt_info, CUSTOM_CLASS_NAMES)
            
            # Run inference on official model
            official_outputs = self.official_rknn.inference(inputs=[input_data])
            all_official_detections = postprocess(official_outputs, scale, self.official_qnt_info, COCO_CLASS_NAMES)
            
            # Filter for person detections only
            person_detections = [det for det in all_official_detections if det['class_name'] == 'person']
            
            # Combine detections
            combined_detections = custom_detections + person_detections
            
            if not self.result_queue.full():
                try: 
                    self.result_queue.put((frame, combined_detections), timeout=0.1)
                except queue.Full: 
                    pass
    
    def _display_loop(self):
        window_name = "船舶实时报警系统"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        audio_manager = AudioManager(self.args.audio_file)
        
        while self.running:
            try:
                original_frame, detections = self.result_queue.get(timeout=2)
                
                # Check for OPEN status and play audio
                if any('OPEN' in d['class_name'] for d in detections): 
                    audio_manager.play()
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 2.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count, self.start_time = 0, time.time()
                
                # Draw and display
                display_frame = draw_results(original_frame, detections, self.fps)
                cv2.imshow(window_name, display_frame)
            except queue.Empty:
                # Show waiting screen
                wait_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(wait_frame, "等待视频流...", (450, 360), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow(window_name, wait_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.running = False
            elif key == ord('f'):  # Toggle fullscreen
                is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN)
        
        audio_manager.cleanup()
        cv2.destroyAllWindows()
    
    def run(self):
        self.capture = VideoCaptureThread(self.args.url)
        self.frame_queue = self.capture.get_frame_queue()
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self.capture.start()
        self.inference_thread.start()
        self._display_loop()
        
        print("正在关闭...")
        self.running = False
        self.capture.stop()
        
        if self.custom_rknn: 
            self.custom_rknn.release()
        if self.official_rknn: 
            self.official_rknn.release()
        
        print("系统已关闭。")

def postprocess(outputs, scale, qnt_info, class_names):
    if not outputs: return []
    output_tensor = outputs[0][0]
    
    # Dequantize if needed
    if qnt_info:
        qnt_type, zp, scale_factor = qnt_info[0]['qnt_type'], qnt_info[0]['zp'], qnt_info[0]['scale']
        float_output = (output_tensor.astype(np.float32) - zp) * scale_factor
    else: 
        float_output = output_tensor.astype(np.float32)
    
    # Transpose if needed
    if float_output.shape[0] < float_output.shape[1]: 
        float_output = float_output.T
    
    boxes, scores, class_ids = [], [], []
    for det in float_output:
        cx, cy, w, h = det[:4]
        class_conf, class_id, conf = det[4:], np.argmax(det[4:]), np.max(det[4:])
        if conf >= 0.25:
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            scores.append(conf)
            class_ids.append(class_id)
    
    if not boxes: return []
    
    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, np.array(scores), 0.20, 0.45)
    final_detections = []
    if indices is not None:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            class_id = int(class_ids[i])
            if class_names and class_id < len(class_names):
                final_detections.append({
                    'bbox': [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)],
                    'confidence': float(scores[i]), 
                    'class_id': class_id, 
                    'class_name': class_names[class_id]
                })
    return final_detections

def draw_results(frame, detections, fps):
    color_map = {
        "hatch_COVER_CLOSE": (0, 255, 0),    # Green
        "hatch_COVER_OPEN": (0, 255, 255),   # Yellow
        "person": (255, 0, 0)                 # Blue
    }
    default_color = (200, 200, 200)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        conf = det['confidence']
        color = color_map.get(class_name, default_color)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw FPS
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generalized Multi-Model Stream Processor")
    parser.add_argument('--url', type=str, required=True, help='RTSP流地址')
    parser.add_argument('--custom_model', type=str, required=True, help='自定义模型路径')
    parser.add_argument('--custom_model_type', type=str, required=True, 
                       choices=['int8', 'fp', 'i16', 'fp16'], help='自定义模型类型')
    parser.add_argument('--official_model', type=str, required=True, help='官方模型路径')
    parser.add_argument('--official_model_type', type=str, required=True, 
                       choices=['int8', 'fp', 'i16', 'fp16'], help='官方模型类型')
    parser.add_argument('--audio_file', type=str, default=None, help='警报音频文件路径')
    args = parser.parse_args()

    processor = None
    
    def cleanup(signum, frame):
        print("\nINFO: 接收到退出信号，正在清理...")
        if processor: 
            processor.running = False
        time.sleep(1.5)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        processor = StreamProcessor(args)
        processor.run()
    except Exception as e:
        print(f"程序主入口发生错误: {e}")
    finally:
        if processor: 
            processor.running = False
        print("程序退出。")
