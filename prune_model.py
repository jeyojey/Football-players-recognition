import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_weights(model, amount=0.2):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module,'weight')
    return model

model_trained = YOLO('/Users/username/PycharmProjects/Football-players-recognition/runs/detect/train/weights/best.pt') #change username

# print the model architecture
model_torch_trained = model_trained.model
print(model_torch_trained)

# prune here
print('model pruning starts now')
prune_weights(model_torch_trained, amount=0.1)
print('model pruned')

print('model saving')
model_trained.save('/Users/username/PycharmProjects/Football-players-recognition/yolo_pruned_new.pt') #change username
print('model saved')