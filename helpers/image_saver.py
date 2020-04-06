import os

from PIL import Image

#current_path = os.getcwd()

def save_image(crop_image, no_of_detected_vehicles):
    crop_image.save('./detected_vehicles/vehicle{}.jpg'.format(no_of_detected_vehicles), 'JPEG')
