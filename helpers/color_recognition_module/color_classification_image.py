import cv2
from .color_recognition_api import color_histogram_feature_extraction
from .color_recognition_api import knn_classifier
import os
import os.path
import sys

def color_recognition(car_img):
    PATH = os.getcwd() + '/helpers/color_recognition_module'
    # read the test image
    '''try:
        source_image = cv2.imread(sys.argv[1])
    except:'''
    prediction = 'n.a.'

    # checking whether the training data is ready

    if os.path.isfile(PATH+'/training.data') and os.access(PATH+'/training.data', os.R_OK):
        print ('training data is ready, classifier is loading...')
    else:
        print ('training data is being created...')
        open('training.data', 'w')
        color_histogram_feature_extraction.training()
        print ('training data is ready, classifier is loading...')

    # get the prediction
    color_histogram_feature_extraction.color_histogram_of_test_image(car_img)
    prediction = knn_classifier.main(PATH+'/training.data', PATH+'/test.data')
    return prediction
