# imports
from ultralytics import YOLO

import torch
# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(cuda_available)

# USE original model
# download model
#model = YOLO('yolov5xu.pt')
# predict
#results = model.predict('input_videos/veo_7.mp4', save=True)#, device='cuda:0')

# USE pruned model
# download model
model = YOLO('yolo_pruned_new.pt')
# predict
results = model.predict('input_videos/veo_7.mp4', save=True)#, device='cuda:0')

# see the prediction
print(results[0])
print('==================')
for box in results[0].boxes:
    print(box)
