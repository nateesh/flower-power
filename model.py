import numpy as np
# import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Dense

# ------
# Task 1  Download the small flower dataset from Blackboard.
# ------

def task1():
    
       
    pass


# ------
# Task 2 Using the tf.keras.applications module download a pretrained MobileNetV2 network.
# ------

def task2():
    import_model = MobileNetV2(
        input_shape=(150, 150, 3),
        alpha=1.0, include_top=False, weights="imagenet",
        input_tensor=None, pooling=None,
        classifier_activation="softmax", #**kwargs
        )
    import_model.trainable = False
    return import_model

# ------
# Task 3 Replace the last layer of the downloaded neural network with a Dense layer of the
# appropriate shape for the 5 classes of the small flower dataset {(x1,t1), (x2,t2),…, (xm,tm)}.
# ------

def task3(import_model):
    x = import_model.layers[-2].output

    outputs = Dense(5, activation="relu", name="flower_power_output_layer")(x)

    model = Model(inputs = import_model.inputs, outputs = outputs)
    
    return model

# ------
# Task 4 Prepare your training, validation and test sets for the non-accelerated version of
# transfer learning.
# ------

def task4():
    
    None

# ----- This is what Fred was talking about in the tutorial
# To rescale an input in the [0, 255] range to be
# in the [0, 1] range, you would pass scale=1./255.
# https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/#rescaling-class

if __name__ == '__main__':
    import_model = task2()
    model = task3(import_model)
    model.summary()