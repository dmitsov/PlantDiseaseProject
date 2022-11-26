import os, shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

TEST_IMG_SIZE = 400
TEST_RAND_ZOOM = 0.1
TEST_RAND_CONTRAST_LOW = 0.1
TEST_RAND_CONTRAST_UPPER = 0.2

def construct_resize_processor(image_height, image_width):
    resize_layers = Sequential([
        layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
        layers.Resizing(TEST_IMG_SIZE, TEST_IMG_SIZE, interpolation='bilinear'),
    ])
    
    return resize_layers

def construct_augment_processor():
    img_augment_layers = Sequential([
        layers.RandomFlip(seed=0),
        layers.RandomRotation(TEST_RAND_ZOOM, seed=0),
        layers.RandomContrast((TEST_RAND_CONTRAST_LOW, TEST_RAND_CONTRAST_UPPER), seed=0)
    ])
    
    return img_augment_layers

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def augment_data_set(dataSet, image_height, image_width):
    resize_layers = construct_resize_processor()
    
    img_augment_layers = construct_augment_processor()
    
    dataSet = dataSet.map(lambda x, y: (resize_layers(x), y), 
              num_parallel_calls= tf.data.AUTOTUNE)
    
    dataSet = dataSet.map(lambda x, y: (img_augment_layers(x), y), num_parallel_calls= tf.data.AUTOTUNE)
    
    return dataSet.cache()