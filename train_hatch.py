import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import os
import subprocess

def train():
    """
    自动执行 YOLOv8 的训练命令。
    """
    # 定义训练命令
    command = [
        "yolo",
        "task=detect",
        "mode=train",
        "model=/best.pt",
        "data=/my_coco.yaml",
        "epochs=300",
        "batch=64",
        "imgsz=640"
    ]
    
    # 将命令列表转换为字符串
    command_str = " ".join(command)
    
    # 打印即将执行的命令
    print(f"Executing command: {command_str}")
    
    # 执行命令
    try:
        subprocess.run(command_str, shell=True, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    train()