
import os
import shutil
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/best.pt")#权重

# Define the output directory
output_dir = "/result"

# Remove the output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed existing output directory: {output_dir}")

# Run inference on the 'test' folder with arguments
model.predict(
    ".jpg",#输入
    classes=0,
    show_labels=True,
    save=True,
    line_width=2,
    project="/new",  # 指定输出的根目录
    name="result"  # 指定在根目录下创建的子文件夹名称
)

print(f"Output saved to: {output_dir}")


# Load a pretrained YOLOv8n model
model2 = YOLO("/best.pt")

# Run inference on 'bus.jpg' with arguments
model2.predict(output_dir, show_labels=True, save=True, line_width=2)


print(f"finish")