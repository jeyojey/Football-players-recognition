# download dataset
from roboflow import Roboflow
rf = Roboflow(api_key="ZEYJn9TghYKQjqYVaiHW")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(12)
dataset = version.download("yolov5")

# check dataset location
print(dataset.location)

# locate dataset
import shutil
shutil.move('football-players-detection-12/train',
            'football-players-detection-12/football-players-detection-12/train')

shutil.move('football-players-detection-12/test',
            'football-players-detection-12/football-players-detection-12/test')

shutil.move('football-players-detection-12/valid',
            'football-players-detection-12/football-players-detection-12/valid')

