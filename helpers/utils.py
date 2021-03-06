import os
import pytz
from uuid import uuid4
from datetime import datetime
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from collections import deque
from PIL import Image, ImageFont, ImageDraw

from .ssd_utils import BBoxUtility
from .image_saver import save_image
from .excel_saver import save_excel
from .color_recognition_module.color_classification_image import color_recognition

current_path = os.getcwd()

class Vehicle:

    def __init__(self, location, Type, n_history=50):
        self._type = Type
        self._location = location
        #self._size = size
        self._detected = True
        self._history = deque([], n_history)

    @property
    def types(self):
        return set(['Car', 'Motorbike', 'Bicycle', 'Bus'])

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._history.append(self._location)
        self._location = location

    def setLocation(self, location):
        self._location = location
        #self._size = size

    def getLocation(self):
        return self._location

class VehicleDetector:
    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    NUM_CLASSES = len(voc_classes) + 1
    _threshold = 0.5
    no_of_detected_vehicles = 0
    detected_vehicles = deque([])

    def __init__(self, model, n_history=50):
        self._model = model
        self._bbox_util = BBoxUtility(self.NUM_CLASSES)
        VehicleDetector.no_of_detected_vehicles = 0
        VehicleDetector.detected_vehicles = deque([])

    @classmethod
    def draw_boxes(cls, img, results, ROI):
        draw_img = Image.fromarray(img)
        width, height = draw_img.size
        draw = ImageDraw.Draw(draw_img, mode='RGBA')
        font = ImageFont.truetype("font/GillSans.ttc", 18)
        padding = 2

        for i in range(6):
            draw.line([(0, ROI+ i*10 ), (width, ROI+ i*10)], fill='red',width=0)

        # Parse the outputs.
        det_label = results[:, 0]
        det_conf = results[:, 1]
        det_xmin = results[:, 2]
        det_ymin = results[:, 3]
        det_xmax = results[:, 4]
        det_ymax = results[:, 5]

        # Get detections with confidence higher than threshold
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= cls._threshold]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        top_indices = [i for i, ymax in enumerate(top_ymax) if (int(round(ymax * img.shape[0])) > ROI and int(round(ymax * img.shape[0])) < ROI+50)]
        top_conf = top_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = top_xmin[top_indices]
        top_ymin = top_ymin[top_indices]
        top_xmax = top_xmax[top_indices]
        top_ymax = top_ymax[top_indices]

        colors = {
            'Car': (255, 128, 0),
            'Multi-axle': (0, 0, 255),
            'Motorbike': (128, 0, 255),
            'Bicycle': (255, 0, 128),
            'Person': (255, 0, 0)
        }

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = cls.voc_classes[label - 1]
            if label_name == 'Bus' or label_name == 'bus':
                label_name = 'Multi-axle'
            display_text = '{} [{:0.2f}]'.format(label_name, score)
            if label_name in set(('Car', 'Multi-axle', 'Motorbike', 'Bicycle')):
                '''Counting logic starts here'''
                if ymax > ROI and ymax < ROI+50:
                    color = colors[label_name]
                    size = draw.textsize(display_text, font)
                    #_draw_rectangle(draw, (xmin, ymin, xmax, ymax), color)
                    draw.rectangle((xmin, ymin, xmax, ymax), outline=color)
                    _draw_rectangle(draw, (xmin, ymin, xmin + size[0] + 2*padding, ymin - size[1] - 2*padding), None, fill=(*color, 40))
                    draw.text((xmin + padding, ymin - size[1] - padding - 2), display_text, (255, 255, 255), font=font)
                    
                    if ymax > ROI:
                        filename = current_path + '/detection-report.xlsx'
                        vehicle_name = 'vehicle-' + str(uuid4())
                        current_datetime = datetime.now(pytz.timezone('Asia/Kolkata'))

                        if VehicleDetector.no_of_detected_vehicles == 0:
                            VehicleDetector.detected_vehicles.append(Vehicle([xmin, ymin, xmax, ymax], label_name))
                            VehicleDetector.no_of_detected_vehicles += 1
                            save_image(((Image.fromarray(img)).crop((xmin, ymin, xmax, ymax))), vehicle_name)
                            detected_color = color_recognition(cv2.imread(current_path+'/detected_vehicles/'+vehicle_name+'.jpg'))
                            save_excel(detections=[[vehicle_name, label_name, detected_color, current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.hour, current_datetime.minute, current_datetime.second]], file_name=filename)
                        else:
                            old_vehicle = False
                            for vehicle in VehicleDetector.detected_vehicles:
                                # print(f'{ymax} , {vehicle.location[3]}') 
                                if abs(ymax - (vehicle.getLocation())[3]) < 10:
                                    vehicle.setLocation([xmin, ymin, xmax, ymax])
                                    #vehicle.accept([xmin, ymin, xmax, ymax])
                                    old_vehicle = True
                                    break
                            if not old_vehicle:
                                if ymax < ROI+20:
                                    VehicleDetector.detected_vehicles.popleft()
                                    VehicleDetector.detected_vehicles.append(Vehicle([xmin, ymin, xmax, ymax], label_name))
                                    VehicleDetector.no_of_detected_vehicles += 1
                                    #print(VehicleDetector.no_of_detected_vehicles)
                                    save_image(((Image.fromarray(img)).crop((xmin, ymin, xmax, ymax))), vehicle_name)
                                    detected_color = color_recognition(cv2.imread(current_path+'/detected_vehicles/'+vehicle_name+'.jpg'))
                                    save_excel(detections=[[vehicle_name, label_name, detected_color, current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.hour, current_datetime.minute, current_datetime.second]], file_name=filename)
                                else:
                                    VehicleDetector.detected_vehicles[-1].location = [xmin, ymin, xmax, ymax]
                            '''Counting logic ends here'''

        # print(len(VehicleDetector.detected_vehicles))
        count_text = 'Count = {}'.format(VehicleDetector.no_of_detected_vehicles)
        size = draw.textsize(count_text, font)
        draw.rectangle((0, 0, size[0] + 2*padding, size[1] + 2*padding), fill=(255, 128, 0))#, 40))
        _draw_rectangle(draw, (0, 0, size[0] + 2*padding, size[1] + 2*padding), None, fill=(255, 128, 0))#, 40))
        draw.text((padding, 0), count_text, (255, 255, 255), font=font)

        return np.asarray(draw_img)

    def detect(self, input_img):
        inputs = []
        img = cv2.resize(input_img, (300, 300))
        img = image.img_to_array(img)
        inputs.append(img.copy())
        inputs = preprocess_input(np.array(inputs))
        inputs = np.expand_dims(inputs[0], axis=0)

        preds = self._model.predict(inputs, batch_size=1, verbose=0)
        results = self._bbox_util.detection_out(preds)

        final_img = self.draw_boxes(input_img, results[0], ROI=200)

        return final_img

    @property
    def pipeline(self):
        def process_frame(input_img):
            return self.detect(input_img)
        return process_frame

def _draw_rectangle(draw, corners, color, fill=None, thickness=3):
    start = -thickness//2
    end = start + thickness
    for i in range(start, end):
        points = [val + i for val in corners]
        draw.rectangle(points, outline=color, fill=fill)