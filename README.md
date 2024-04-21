# CarlaRealtime_ImageClassification
The aim of this script is to provide a template code to perform realtime image classification with Tensorflow.
Here, the dynamic environment is simulated by Carla Simulator (0.9.15) and the object detection model trained on the 
Coco dataset is fetched from TensorFlow hub (https://www.kaggle.com/models?publisher=tensorflow&tfhub-redirect=true). 

The script connects to carla server and spawns an ego vehicle and an RGB camera attached to it in the virtual world. 
Then, further vehicles, i.e., 30 are spawned at random in the scene. The object detection model is then loaded from 
TensorFlow hub and run to detect the objects in the field of view of ego vehicle's RGB camera.

A window shows the realtime image classification as seen by the camera.

## Usage
Once the carla server is up an running, the realtime classification is performed by running the command:
```bash
python3 carlaRealTime_ImageClassification.py
```

## Python version and packages
The code was written and tested with Python 3.10.12, TensorFlow GPU, and Carla 0.9.15 on docker. To run the code, 
the following python packages are required:
- carla
- numpy
- seaborn
- opencv
- tensorflow
- tensorflow-hub
- object detection api

## Label map
The detected objects are labeled based on the mscoco label map provided in the labels folder as .*pbtxt file.

## Acknowledgment
Thank you to all the people who shared pieces of codes which inspired this script. Feel free to use, 
share, and modify as you wish.

Any feedback will be highly appreciated :)
