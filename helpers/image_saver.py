import os

from PIL import Image

current_path = os.getcwd()

def save_image(crop_image, name):
    crop_image.save(current_path+'/detected_vehicles/{}.jpg'.format(name), 'JPEG')
