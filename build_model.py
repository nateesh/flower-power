import pathlib
import os

import numpy as np
# import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, utils, optimizers, losses


IMG_SIZE = (256, 256)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
flowers_dir = 'small_flower_dataset/'
flower_labels = sorted(os.listdir(flowers_dir))
print(flower_labels)

def task_1():
    """
    Task 1: Download the small flower dataset from Blackboard (DONE)
    """
    pass

def task_2():
    """
    Task 2: Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    Input: None
    Output: a freeze base model
    """
    base_model = MobileNetV2(
        input_shape=IMG_SHAPE,
        alpha=1.0, include_top=False, weights="imagenet",
        input_tensor=None, pooling=None,
        classifier_activation="softmax", #**kwargs
        )
    base_model.trainable = False
    return base_model

def task_3(base_model):
    """
    # Task 3 - Replace the last layer of the downloaded neural network with a Dense layer of the
    # appropriate shape for the 5 classes of the small flower dataset {(x1,t1), (x2,t2),…, (xm,tm)}.
    Input: a freeze base model
    Output: a model with new layer on top
    """
    x = base_model.layers[-2].output
    
    # A Denset layer of 5 classes
    outputs = layers.Dense(5, activation="relu", name="flower_power_layer")(x)
    
    model = Model(inputs = base_model.inputs, outputs = outputs)
    
    return model

if __name__ == '__main__':
    import_model = task_2()
    model = task_3(import_model).summary()