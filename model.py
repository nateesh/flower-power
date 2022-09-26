import numpy as np
# import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, utils, optimizers, losses


# ------
# Task 1 - Download the small flower dataset from Blackboard.
# ------

def task1():



    pass

def task2():
    """
    Task 2 - Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    """
    import_model = MobileNetV2(
        input_shape=(256, 256, 3),
        alpha=1.0, include_top=False, weights="imagenet",
        input_tensor=None, pooling=None,
        classifier_activation="softmax", #**kwargs
        )
    import_model.trainable = False
    return import_model

def task3(import_model):
    """
    # Task 3 - Replace the last layer of the downloaded neural network with a Dense layer of the
    # appropriate shape for the 5 classes of the small flower dataset {(x1,t1), (x2,t2),…, (xm,tm)}.
    """
    x = import_model.layers[-2].output
    
    ## ---- !!!! ---- !!!! ---- !!!! ---- !!!! ---- !!!!
    outputs = layers.Dense(5, activation="relu", name="flower_power_output_layer")(x)
    
    # model is having trouble compiling in step 5 because of its shape should be (None, 5) not (None, 8, 8, 5):
    # "flower_power_output_layer (Dense  (None, 8, 8, 5)     6405        ['Conv_1_bn[0][0]']"
    ## ---- !!!! ---- !!!! ---- !!!! ---- !!!! ---- !!!!
    
    model = Model(inputs = import_model.inputs, outputs = outputs)
    
    return model



## recales example https://keras.io/guides/preprocessing_layers/

def task4():
    """
    Task 4 - Prepare your training, validation and test sets for the non-accelerated version of
    transfer learning.
    """
    
    # using https://keras.io/examples/vision/image_classification_from_scratch/
    # https://www.tensorflow.org/tutorials/images/classification
    
    IMAGE_SIZE = (256, 256)
    directory_path = "flower_dataset/small_flower_dataset"
    
    train_ds = utils.image_dataset_from_directory(
        directory=directory_path,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        color_mode="rgb",
        shuffle=True,
        seed=2, # same seed for both X and y datasets to avoid overlap, could be any number but must be same
        validation_split=0.2, # portion (%) reserved for validation
        subset="training" # portion to be assignment to X_dataset
        )
    
    val_ds = utils.image_dataset_from_directory(
        directory=directory_path,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        color_mode="rgb",
        shuffle=True,
        seed=2,
        validation_split=0.2,
        subset="validation"
        )
    
    # ------- Plot sample of images
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")
    # plt.show()
    
    # --- Resacle the training dataset (q: what about test dataset)
    rescaling_layer = layers.Rescaling(scale=1.0 / 255)
    train_ds_rescaled = train_ds.map(lambda x, y: (rescaling_layer(x), y))

    return train_ds_rescaled, val_ds


def task5(model, train_ds, val_ds):
    """
    # Task 5 - Compile and train your model with an SGD3 optimizer using the 
    # following parameters learning_rate=0.01, momentum=0.0, nesterov=False.
    
    https://keras.io/examples/vision/image_classification_from_scratch/
    
    """
    
    # "use buffered prefetching so we can yield data 
    # from disk without having I/O becoming blocking"    
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    
    epochs = 50
    
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
        loss="categorical_crossentropy",
    )
    
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)



if __name__ == '__main__':
    import_model = task2()
    model = task3(import_model)
    model.summary()
    # train_ds, val_ds = task4()
    # task5(model, train_ds, val_ds)
    
    