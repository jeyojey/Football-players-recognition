#!yolo task=detect mode=train model=yolov5xu.pt data={dataset.location}/data.yaml epochs=10 imgsz=640

import os
# Define the path to the YOLOv5 model and dataset
model_path = 'yolov5xu.pt'
dataset_location = '/Users/username/PycharmProjects/Football-players-recognition/football-players-detection-12'  # change username
data_yaml = f'{dataset_location}/data.yaml'
epochs = 2
imgsz = 640

# Build the YOLOv5 training command
command = f'yolo task=detect mode=train model={model_path} data={data_yaml} epochs={epochs} imgsz={imgsz}'

# Run the command using os.system
os.system(command)
