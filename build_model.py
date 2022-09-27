import pathlib
import os
import datetime
import time

import numpy as np
# import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, utils, optimizers, losses
import tensorflow as tf
import matplotlib.pyplot as plt


# model expected shape=(None, 224, 224)
IMG_SIZE = (224, 224)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
flowers_dir = 'small_flower_dataset/'




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
        input_shape=(224, 224, 3),
        alpha=1.0, include_top=True, weights="imagenet",
        input_tensor=None, pooling=None,
        classifier_activation="softmax"
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
    flower_output = base_model.layers[-2].output
    
    # A Denset layer of 5 classes
    outputs = layers.Dense(5, activation="relu", name="flower_power_layer")(flower_output)
    
    model = Model(inputs = base_model.inputs, outputs = outputs)
    
    return model

def task_4():
    """
    Task 4 - Prepare your training, validation and test sets for the non-accelerated version of
    transfer learning.
    """
    batch_size = 32
    train_ds = tf.keras.utils.image_dataset_from_directory(
                directory = flowers_dir,
                labels="inferred",
                label_mode="int",
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=IMG_SIZE,
                batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
                directory = flowers_dir,
                labels="inferred",
                label_mode="int",
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=IMG_SIZE,
                batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Standardize the data
    # The RGB channel values are in the [0, 255] range. 
    # This is not ideal for a neural network; in general you should seek to make your input values small.
    normalization_layer = layers.Rescaling(1./255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_ds, normalized_val_ds

def task_5(flower_model, train_ds, val_ds):
    """
    # Task 5 - Compile and train your model with an SGD3 optimizer using the 
    # following parameters learning_rate=0.01, momentum=0.0, nesterov=False.
    """
    epochs = 5

    # To freeze a layer, simply set its trainable property to False.
    #  We do this for all layers except the last one, which is our newly created output layer.
    for layer in flower_model.layers[:-1]:
        layer.trainable = False
    
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    start = time.time()
    # Train model
    flower_model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    history = flower_model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    # end time
    end = time.time()
    print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print ("[STATUS] duration: {}".format(end - start))
    

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']


    epochs_range = range(20)
    plt.figure(figsize=(8, 8))



    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')



    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == '__main__':
    import_model = task_2()
    flower_model = task_3(import_model)
    flower_model.summary()
    train_ds, val_ds = task_4()
    task_5(flower_model, train_ds, val_ds)