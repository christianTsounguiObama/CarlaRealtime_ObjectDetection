#!/usr/bin/env python3

'''
    The aim of this script is to provide a template code to perform realtime object detection with Tensorflow.
Here, the dynamic environment is simulated by Carla Simulator (0.9.15) and the object detection model trained on the 
Coco dataset is fetched from TensorFlow hub (https://www.kaggle.com/models?publisher=tensorflow&tfhub-redirect=true). 

The script connects to carla server and spawns an ego vehicle and an RGB camera attached to it in the virtual world. 
Then, further vehicles, i.e., 30 are spawned at random in the scene. The object detection model is then loaded from 
TensorFlow hub and run to detect the objects in the field of view of ego vehicle's RGB camera.

A window shows the realtime object detection as seen by the camera.

Usage: python3 carlaRealTime_ObjectDetection.py
'''

import carla
import random 
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import label_map_util
import seaborn as sns

class CarlaInit():
    '''
        This class contains the methods needed to connect to the carla server,
        spawn vehicles and the ego camera, and set the vehicles to carla default 
        autopilot mode.
    '''
    def __init__(self, port):
        self.client = carla.Client('localhost', port)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

    def spawnActors(self, carName, numberOfActors):
        '''
            This function spawns the ego vehicle with name given by the parameter 'CarName',
            as well as further 'numberOfActors' vehicles in the world.

            carName = Ego vehicle name as in the carla blue prints library
            numberOfActors: Number of actor vehicles to spawn in the world
        '''
        self.ego = self.bp_lib.find(carName)
        self.ego = self.world.try_spawn_actor(self.ego, random.choice(self.spawn_points))
        for i in range(numberOfActors):
            self.actor = random.choice(self.bp_lib.filter('vehicle'))
            _ = self.world.try_spawn_actor(self.actor, random.choice(self.spawn_points))

    def spawnSensor(self, sensorName):
        '''
            This function spawns a sensor and attaches it to the ego vehicle.
            In this case the sensor is an RGB camera.

            sensorName = sensor name as in the carla blue prints library
        '''
        self.sensorBlueprint = self.bp_lib.find(sensorName)
        self.cameraTrans = carla.Transform(carla.Location(z=2))
        self.sensor = self.world.try_spawn_actor(self.sensorBlueprint, self.cameraTrans, attach_to=self.ego)

    def callback(self, image):
        '''
            This function  collects the images from the RGB camera and saves them
            in a global dictionary for further processing with the object detection
            model.

            image = image from RGB camera
        '''
        global dataDict
        dataDict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    def readSensor(self):
        '''
            This function listens to the RGB camera and calls the callback function
            each time an image is availble.
        '''
        self.sensor.listen(lambda image: self.callback(image))

    def launchCarla(self, carname, sensorName):
        '''
            This function spawns the ego vehicle alongside 30 vehicles in the world.
            Then it spawns the RGB camera on the ego vehicle and collects the pictures
            from the camera.
        '''
        self.spawnActors(carname, 30)
        self.spawnSensor(sensorName)
        self.readSensor()

    def setActorstoAutopilot(self):
        '''
            This function sets all vehicles in autopilot mode.
        '''
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.set_autopilot(True)

class ImageClassification():
    def __init__(self, threshold, tfhubModel, labelMapPath):
        tf.keras.backend.clear_session()
        self._threshold = threshold
        self.hubmodel = hub.load(tfhubModel)
        print("model loaded")
        self._labelMapPath = labelMapPath
        self.categoryIndex = label_map_util.create_category_index_from_labelmap(self._labelMapPath, use_display_name=True)
        self.colors = sns.color_palette(None, len(self.categoryIndex))
        self.results = {}
        self.processedResults = {}

        self.carlainit = CarlaInit(2000)
        print("Connection to carla server established")

    def classify(self, image):
        imWidth, imHeight, _ = np.asarray(image).shape
        imageTensor = np.asarray(tf.convert_to_tensor(image[:,:,:3], dtype=tf.uint8)).reshape((1, imWidth, imHeight, 3)).astype(np.uint8)
        results = self.hubmodel(imageTensor)
        self.results = {key:value.numpy() for key,value in results.items()}
    
    def process(self):
        detections = self.results['detection_scores'] >= self._threshold
        scores = self.results['detection_scores'][detections]
        classes = self.results['detection_classes'][detections]
        boxes = self.results['detection_boxes'][detections,:]
        self.processedResults = {'num_detections':np.sum(detections), 'classes':classes, 'scores':scores, 'boxes':boxes}
    
    def visualize(self, image):
        height, width = image.shape[0], image.shape[1]
        read = 0
        for class_index in self.processedResults['classes']:
            if int(class_index) > len(self.categoryIndex):
                break

            # boxes coordinates
            upper_left_x = int(self.processedResults['boxes'][read,1] * width)
            upper_left_y = int(self.processedResults['boxes'][read,0] * height)
            lower_right_x = int(self.processedResults['boxes'][read,3] * width)
            lower_right_y = int(self.processedResults['boxes'][read,2] * height)
            predict_text = self.categoryIndex[int(class_index)]['name'] + ": " + str(np.round(self.processedResults['scores'][read], 2))
            read += 1

            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), np.array(self.colors[int(class_index)])*255.0, 2)
            cv2.putText(image, "{}".format(predict_text), (upper_left_x, upper_left_y - 10 if upper_left_y > 30 else upper_left_y + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, np.array(self.colors[int(class_index)])*255.0, 1)
        cv2.imshow('Detections', image)

    def launchImageClassification(self):
        global dataDict
        framerate = 200
        wait = int(1000/framerate)
        self.carlainit.setActorstoAutopilot()
        while True:
            image = dataDict['image']
            self.classify(image)
            self.process()
            self.visualize(image)
            if cv2.waitKey(wait) == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    dataDict = {}
    carlainit = CarlaInit(2000)
    carlainit.launchCarla('vehicle.lincoln.mkz_2020', 'sensor.camera.rgb')

    detectionThreshold = 0.50
    tfhubModel = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'
    labelMapPath = '/labels/mscoco_label_map.pbtxt' # Specify the full path to the labels folder
    onlineImageclassifier = ImageClassification(detectionThreshold, tfhubModel, labelMapPath)
    onlineImageclassifier.launchImageClassification()
