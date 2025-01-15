

# Football players recognition application using YOLO with finetuned and prunned weights

This toy project implements the football players recognition on a video using YOLO models. The model is first uploaded with inference applied. The code for inference is located in `inference.py`.
The model is then finetuned using the [football-players-detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) from roboflow.com. To download the dataset use `download_dataset.py`. Training part is located in `train_yolo.py`. After training the recognition of 'person' is substituted by 'player'.


Finally, the finetuned model is pruned with the standard l1 algorithm to reduce the total complexity. The code for pruning the best model is located in `prune_model.py`.

The model was trained on GPU in Google Colaboratory.

# Demo
Original video: `input_videos/veo_7.mp4`

Video obtained after YOLO recognition without finetuning: `runs/detect/predict1/veo_7.mp4`

After pruning video: `runs/detect/predict3/veo_7.mp4`




# Links
YOLO model used: https://docs.ultralytics.com/models/yolov5/

Dataset used: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
