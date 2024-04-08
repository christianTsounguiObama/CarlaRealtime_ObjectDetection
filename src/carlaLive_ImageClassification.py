import carla
import random 
import numpy as np
import cv2
from six import BytesIO
from PIL import Image
from six.moves.urllib.request import urlopen
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import label_map_util
import seaborn as sns

class CarlaInit():
    def __init__(self, port):
        self.client = carla.Client('localhost', port)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

    def spawnActors(self, carName, numberOfActors):
        self.ego = self.bp_lib.find(carName)
        self.ego = self.world.try_spawn_actor(self.ego, random.choice(self.spawn_points))
        for i in range(numberOfActors):
            self.actor = random.choice(self.bp_lib.filter('vehicle'))
            _ = self.world.try_spawn_actor(self.actor, random.choice(self.spawn_points))

    def spawnSensor(self, sensorName):
        self.sensorBlueprint = self.bp_lib.find(sensorName)
        self.cameraTrans = carla.Transform(carla.Location(z=2))
        self.sensor = self.world.try_spawn_actor(self.sensorBlueprint, self.cameraTrans, attach_to=self.ego)

    def callback(self, image):
        global dataDict
        dataDict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    def readSensor(self):
        #global dataDict
        #imageWidth = self.sensorBlueprint.get_attribute("image_size_x").as_int()
        #imageHeight = self.sensorBlueprint.get_attribute("image_size_y").as_int()
        #sensorData = {'image': np.zeros((imageHeight, imageWidth, 4))}
        self.sensor.listen(lambda image: self.callback(image))

    def launchCarala(self, carname, sensorName):
        self.spawnActors(carname, 30)
        self.spawnSensor(sensorName)
        self.readSensor()

    def setActorstoAutopilot(self):
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
    carlainit.launchCarala('vehicle.lincoln.mkz_2020', 'sensor.camera.rgb')

    detectionThreshold = 0.65
    tfhubModel = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'
    labelMapPath = '/home/christian/Documents/Artifitial_Intelligence/ADAS/models/research/object_detection/data/mscoco_label_map.pbtxt'
    onlineImageclassifier = ImageClassification(detectionThreshold, tfhubModel, labelMapPath)
    onlineImageclassifier.launchImageClassification()
