import datetime
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model, optimizers, losses
import matplotlib.pyplot as plt

# model expected shape=(None, 224, 224)
IMG_SIZE = (224, 224)
flowers_dir = 'small_flower_dataset/'
EPOCHS = 30

""" Project Team

    - Phan Thao Nhi Nguyen n11232862
	- Nathan Eshraghi - n11353295
 
"""

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
    # import MobileNetV2 base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights="imagenet")

    # Freeze layer exclude new layer
    for layer in base_model.layers:
        layer.trainable=False
    return base_model

def task_3(base_model):
    """
    # Task 3 - Replace the last layer of the downloaded neural network with a Dense layer of the
    # appropriate shape for the 5 classes of the small flower dataset {(x1,t1), (x2,t2),…, (xm,tm)}.
    Input: a freeze base model
    Output: a model with new layer on top
    """

    # remove the classification layer
    x = base_model.layers[-2].output

    # add new layers to accomodate classification of new small flower dataset
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    model = Model(inputs = base_model.inputs, outputs = outputs)

    return model

def task_4():
    """
    Task 4 - Prepare your training, validation and test sets for the non-accelerated version of
    transfer learning.
    """
    batch_size = 32

    # Split training, and no training images into two datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
                flowers_dir,
                labels='inferred',
                label_mode='int',
                class_names=None,
                color_mode='rgb',
                batch_size=batch_size,
                image_size=IMG_SIZE,
                shuffle=True,
                seed=2,
                validation_split=0.3,
                subset="training",
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False)

    val_ds = tf.keras.utils.image_dataset_from_directory(
                flowers_dir,
                labels='inferred',
                label_mode='int',
                class_names=None,
                color_mode='rgb',
                batch_size=batch_size,
                image_size=IMG_SIZE,
                shuffle=True,
                seed=2,
                validation_split=0.3,
                subset="validation",
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False)

    # further split the non-training dataset into two datasets
    # one for validation, the other for testing
    test_ds = val_ds.take(5)
    val_ds = val_ds.skip(5)

    print(f"Train ds: {len(train_ds)} batches")
    print(f"Test ds : {len(test_ds)} bacthes")
    print(f"Val ds: {len(val_ds)} batches")

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Standardize the data
    # The RGB channel values are in the [0, 255] range.
    # Create a rescaling layer to scale values between -1 and 1, consistent with
    # the tf.keras.applications.mobilenet_v2.preprocess_input method
    normalization_layer = layers.Rescaling(scale=1./127.5, offset=-1)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_ds, normalized_val_ds, normalized_test_ds

def task_5(flower_model, train_ds, val_ds):
    """
    Task 5 - Compile and train your model with an SGD3 optimizer using the
    following parameters learning_rate=0.01, momentum=0.0, nesterov=False.
    Input: flower model with new layer, training dataset, validation dataset
    Output: a history object that is a record of training loss values,
            validation loss values.
    """

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    start = time.time()

    # Train model with lr 0,01, momentum 0
    # Declare learning rate constant
    LR_1 = 0.01
    
    # Declare momentum constant
    MOMENTUM = 0

    # Compile the model
    flower_model.compile(
        optimizer=optimizers.SGD(learning_rate=LR_1, momentum=MOMENTUM, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # fit the model and store training history
    history = flower_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    # end time
    end = time.time()
    print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print ("[STATUS] duration: {}".format(end - start))

    return history

def task_6(history):
    '''
    Task 6: Plot the training and validation errors vs time as well as the training and validation
            accuracies.
    Input: history object (from Model.fit())
    Output: training model with learning_rate=0.01, momentum=0.0 val_loss, val_accuracy graph

    '''
    print("starting plot")

    # Define variables to plot
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs_range = range(EPOCHS)

    # plot the model training accuracy and loss over epochs
    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.ylim(0.1, 1.1)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylim(-0.1, 2)

    plt.show()
    print("plot finished")

def flatten_list(l):
    """ Used to find consistent y axis limits for graph plots.
    Input: a list of lists
    Output: a single flattened list
    """
    return [val for sublist in l for val in sublist]

def task_7(import_model, train_ds, val_ds):
    """
    Task 7: Experiment with 3 different orders of magnitude for the learning rate (0.001, 0.1, 1). Plot the
            results, draw conclusions.
    Input: model, trainning and validate dataset
    Output: plot model result (3 learning rates) on 3 graphs

    """

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    # Establish learning rates
    LR_1 = 0.001
    LR_ORIG = 0.01
    LR_2 = 0.1
    LR_3 = 1

    # Store model in variable
    learn_rate_1 = task_3(import_model)
    learn_rate_original = task_3(import_model)
    learn_rate_2 = task_3(import_model)
    learn_rate_3 = task_3(import_model)

    # Train model with 0.001 learning rate
    learn_rate_1.compile(
        optimizer=optimizers.SGD(learning_rate=LR_1, momentum=0.0, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    # Train model with 0.01 learning rate
    learn_rate_original.compile(
        optimizer=optimizers.SGD(learning_rate=LR_ORIG, momentum=0.0, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    
    # Train model with 0.1 learning rate
    learn_rate_2.compile(
        optimizer=optimizers.SGD(learning_rate=LR_2, momentum=0.0, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    # Train model with 1 learning rate
    learn_rate_3.compile(
        optimizer=optimizers.SGD(learning_rate=LR_3, momentum=0.0, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    epochs_range = range(EPOCHS)

    # fit models and store training history
    lr_1_history = learn_rate_1.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    lr_orig_history = learn_rate_original.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    lr_2_history = learn_rate_2.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    lr_3_history = learn_rate_3.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    # Declare plotting variables for each test case
    lr_1_loss = lr_1_history.history['loss']
    lr_1_val_loss = lr_1_history.history['val_loss']
    lr_1_acc = lr_1_history.history['accuracy']
    lr_1_val_acc = lr_1_history.history['val_accuracy']

    lr_orig_loss = lr_orig_history.history['loss']
    lr_orig_val_loss = lr_orig_history.history['val_loss']
    lr_orig_acc = lr_orig_history.history['accuracy']
    lr_orig_val_acc = lr_orig_history.history['val_accuracy']
    
    lr_2_loss = lr_2_history.history['loss']
    lr_2_val_loss = lr_2_history.history['val_loss']
    lr_2_acc = lr_2_history.history['accuracy']
    lr_2_val_acc = lr_2_history.history['val_accuracy']

    lr_3_loss = lr_3_history.history['loss']
    lr_3_val_loss = lr_3_history.history['val_loss']
    lr_3_acc = lr_3_history.history['accuracy']
    lr_3_val_acc = lr_3_history.history['val_accuracy']

    # find the y axis limits for the plots
    loss_vals = [lr_1_loss, lr_1_val_loss, lr_2_loss, lr_2_val_loss, lr_3_loss, lr_3_val_loss, lr_orig_loss, lr_orig_val_loss]
    loss_y_axis_min = min(flatten_list(loss_vals)) - 0.1

    accuracies = [lr_1_acc, lr_1_val_acc, lr_2_acc, lr_2_val_acc, lr_3_acc, lr_3_val_acc, lr_orig_acc, lr_orig_val_acc]
    acc_y_axis_max = max(flatten_list(accuracies)) + 0.1
    acc_y_axis_min = min(flatten_list(accuracies)) - 0.1
    
    # Make the plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 4, 1)
    plt.plot(epochs_range, lr_1_acc, label='Training')
    plt.plot(epochs_range, lr_1_val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.title(f'Learning Rate = {LR_1}')
    plt.ylabel(f'Accuracy', fontsize=12)
    plt.ylim(acc_y_axis_min, acc_y_axis_max)

    plt.subplot(2, 4, 2)
    plt.plot(epochs_range, lr_orig_acc, label='Training')
    plt.plot(epochs_range, lr_orig_val_acc, label='Validation')
    plt.title(f'Learning Rate = {LR_ORIG}')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)

    plt.subplot(2, 4, 3)
    plt.plot(epochs_range, lr_2_acc, label='Training')
    plt.plot(epochs_range, lr_2_val_acc, label='Validation')
    plt.title(f'Learning Rate = {LR_2}')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)

    plt.subplot(2, 4, 4)
    plt.plot(epochs_range, lr_3_acc, label='Training')
    plt.plot(epochs_range, lr_3_val_acc, label='Validation')
    plt.title(f'Learning Rate = {LR_3}')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)

    plt.subplot(2, 4, 5)
    plt.plot(epochs_range, lr_1_loss, label='Training')
    plt.plot(epochs_range, lr_1_val_loss, label='Validation')
    plt.ylim(loss_y_axis_min, 2)
    plt.ylabel(f'Loss', fontsize=12)

    plt.subplot(2, 4, 6)
    plt.plot(epochs_range, lr_orig_loss, label='Training')
    plt.plot(epochs_range, lr_orig_val_loss, label='Validation')
    plt.ylim(loss_y_axis_min, 2)
    
    plt.subplot(2, 4, 7)
    plt.plot(epochs_range, lr_2_loss, label='Training')
    plt.plot(epochs_range, lr_2_val_loss, label='Validation')
    plt.ylim(loss_y_axis_min, 2)

    plt.subplot(2, 4, 8)
    plt.plot(epochs_range, lr_3_loss, label='Training')
    plt.plot(epochs_range, lr_3_val_loss, label='Validation')

    plt.show()

def task_8(import_model, train_ds, val_ds, test_ds):
    """
    Task 8: Choose the beast learning rate from task 8 and training model with non zero momentum (0.5).
            For comparing in this task, we trained 2 model with different momentum:
            model 1: lr = 0.01, momentum = 0.0
            model 2: lr = 0.01, momentum = 0.5
    Input: model, trainning dataset, validate dataset, testing dataset
    Output: graph with different momentum
    """

    # the best learning rate found in task 7
    LEARNING_RATE = 0.1

    # declare 3 non-zero momentums
    MOMENTUM_0 = 0
    MOMENTUM_1 = 0.15
    MOMENTUM_2 = 0.4
    MOMENTUM_3 = 0.8

    m0_model = task_3(import_model)
    m1_model = task_3(import_model)
    m2_model = task_3(import_model)
    m3_model = task_3(import_model)

    # Train model with momentum = 0 (the previous task_7 selection for learning rate)
    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_0} momentum model TRAINING')
    m0_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_0, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m0_history = m0_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    m0_eval_results = m0_model.evaluate(test_ds)
    
    # Train 3 models with different momentums
    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_1} momentum model TRAINING')
    m1_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_1, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m1_history = m1_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    m1_eval_results = m1_model.evaluate(test_ds)

    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_2} momentum model TRAINING')
    m2_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_2, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m2_history = m2_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    m2_eval_results = m2_model.evaluate(test_ds)

    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_3} momentum model TRAINING')
    m3_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_3, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m3_history = m3_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    m3_eval_results = m3_model.evaluate(test_ds)

    # Prepare plot argument for model lr = 0.1, momentum_0
    m0_loss = m0_history.history['loss']
    m0_val_loss = m0_history.history['val_loss']
    m0_acc = m0_history.history['accuracy']
    m0_val_acc = m0_history.history['val_accuracy']
    
    # Prepare plot argument for model lr = 0.1, momentum_1
    m1_loss = m1_history.history['loss']
    m1_val_loss = m1_history.history['val_loss']
    m1_acc = m1_history.history['accuracy']
    m1_val_acc = m1_history.history['val_accuracy']

    # Prepare plot argument for model lr = 0.1, momentum_2
    m2_loss = m2_history.history['loss']
    m2_val_loss = m2_history.history['val_loss']
    m2_acc = m2_history.history['accuracy']
    m2_val_acc = m2_history.history['val_accuracy']

    # Prepare plot argument for model lr = 0.1, momentum_3
    m3_loss = m3_history.history['loss']
    m3_val_loss = m3_history.history['val_loss']
    m3_acc = m3_history.history['accuracy']
    m3_val_acc = m3_history.history['val_accuracy']

    acc_vals = [m0_acc, m0_val_acc, m1_acc, m1_val_acc, m2_acc, m2_val_acc, m3_acc, m3_val_acc]
    loss_vals =[m0_loss, m0_val_loss, m1_loss, m1_val_loss, m2_loss, m2_val_loss, m3_loss, m3_val_loss]

    acc_y_axis_max = max(flatten_list(acc_vals)) + 0.1
    acc_y_axis_min = min(flatten_list(acc_vals)) - 0.1
    loss_y_axis_max = max(flatten_list(loss_vals)) + 0.1
    loss_y_axis_min = min(flatten_list(loss_vals)) - 0.1

    epochs_range = range(EPOCHS)

    # plot the loss and accuracy for different momentums
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 4, 1)
    plt.plot(epochs_range, m0_acc, label='Training')
    plt.plot(epochs_range, m0_val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_0}')

    plt.subplot(2, 4, 2)
    plt.plot(epochs_range, m1_acc, label='Training Accuracy')
    plt.plot(epochs_range, m1_val_acc, label='Validation Accuracy')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_1}')
    
    plt.subplot(2, 4, 3)
    plt.plot(epochs_range, m2_acc, label='Training Accuracy')
    plt.plot(epochs_range, m2_val_acc, label='Validation Accuracy')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_2}')

    plt.subplot(2, 4, 4)
    plt.plot(epochs_range, m3_acc, label='Training Accuracy')
    plt.plot(epochs_range, m3_val_acc, label='Validation Accuracy')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_3}')

    plt.subplot(2, 4, 5)
    plt.plot(epochs_range, m0_loss, label='Training Loss')
    plt.plot(epochs_range, m0_val_loss, label='Validation Loss')
    plt.ylabel(f'Loss', fontsize=12)
    plt.xlabel(f"\nTest ds loss: {int(round(m0_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m0_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.subplot(2, 4, 6)
    plt.plot(epochs_range, m1_loss, label='Training Loss')
    plt.plot(epochs_range, m1_val_loss, label='Validation Loss')
    plt.ylabel(f'Loss', fontsize=12)
    plt.xlabel(f"\nTest ds loss: {int(round(m1_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m1_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.subplot(2, 4, 7)
    plt.plot(epochs_range, m2_loss, label='Training Loss')
    plt.plot(epochs_range, m2_val_loss, label='Validation Loss')
    plt.xlabel(f"\nTest ds loss: {int(round(m2_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m2_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.subplot(2, 4, 8)
    plt.plot(epochs_range, m3_loss, label='Training Loss')
    plt.plot(epochs_range, m3_val_loss, label='Validation Loss')
    plt.xlabel(f"\nTest ds loss: {int(round(m3_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m3_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.show()

def task_9():
    '''
    Task 9: Prepare your training, validation and test sets. Those are based on {(F(x1).t1),
        (F(x2),t2),…,(F(xm),tm)}
    Input: model, trainning dataset, validate dataset, testing dataset
    Output: new training, validation and test sets with accelerated
    '''

    # Prepare for training, validating, testing dataset
    batch_size = 32
    train_ds = tf.keras.utils.image_dataset_from_directory(
                    flowers_dir,
                    labels='inferred',
                    label_mode='int',
                    class_names=None,
                    color_mode='rgb',
                    batch_size=batch_size,
                    image_size=IMG_SIZE,
                    shuffle=True,
                    seed=2,
                    validation_split=0.3,
                    subset="training",
                    interpolation='bilinear',
                    follow_links=False,
                    crop_to_aspect_ratio=False)
    val_ds = tf.keras.utils.image_dataset_from_directory(
                    flowers_dir,
                    labels='inferred',
                    label_mode='int',
                    class_names=None,
                    color_mode='rgb',
                    batch_size=batch_size,
                    image_size=IMG_SIZE,
                    shuffle=True,
                    seed=2,
                    validation_split=0.3,
                    subset="validation",
                    interpolation='bilinear',
                    follow_links=False,
                    crop_to_aspect_ratio=False)

    testing_ds = val_ds.take(5)
    val_ds = val_ds.skip(5)

    print(f"Train ds length: {len(train_ds)} batches")
    print(f"Test ds length: {len(testing_ds)} bacthes")
    print(f"Val length: {len(val_ds)} batches ")


    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Standardize the data
    # The RGB channel values are in the [0, 255] range.
    # Create a rescaling layer to scale values between -1 and 1, consistent with
    # the tf.keras.applications.mobilenet_v2.preprocess_input method
    normalization_layer = layers.Rescaling(scale=1./127.5, offset=-1)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_testing_ds = testing_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = normalized_train_ds
    val_ds = normalized_val_ds
    testing_ds = normalized_testing_ds


    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    
    # Freeze layer exclude new layer
    for layer in base_model.layers:
        layer.trainable=False

    # Feature extraction
    feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.output)

    train_activations = feature_extractor.predict(train_ds)
    val_activations = feature_extractor.predict(val_ds)
    test_activations = feature_extractor.predict(testing_ds)

    # Declare pooling layer and apply to activations
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    train_x_avg = global_average_layer(train_activations)
    val_x_avg = global_average_layer(val_activations)
    test_x_avg = global_average_layer(test_activations)

    # Checking for feature_extractor summary
    feature_extractor.summary()
    
    # Extract the labels from the _ds batches and concatenate them into one array
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    val_labels = np.concatenate([y for x, y in val_ds], axis=0)
    test_labels = np.concatenate([y for x, y in testing_ds], axis=0)

    return train_x_avg, train_labels, val_x_avg, val_labels, test_x_avg, test_labels

def task_10(train_x_avg, train_labels, val_x_avg, val_labels, test_x_avg, test_labels):
    '''
    Task 10: Perform new training, validation and test sets (task 9) on model with
            best learning rate and non-zero momentum.
            For comparing in this task, we trained 2 model with different momentum:
            model 1: lr = 0.01, momentum = 0.0
            model 2: lr = 0.01, momentum = 0.5
    Input: training, validation and test sets (task 9)
    '''

    # Create model input layer and classification layers
    inputs = tf.keras.Input(shape=(1280), name="Extracted Features")
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(5, activation='softmax', name="Classifier")(x)

    # Declare model variables for each momentum being tested
    m1_model = Model(inputs = inputs, outputs = outputs)
    m2_model = Model(inputs = inputs, outputs = outputs)
    m3_model = Model(inputs = inputs, outputs = outputs)

    # Perform with the best learning rate and 3 non-zero momentums
    # The best learning rate found in task 7
    LEARNING_RATE = 0.1

    MOMENTUM_1 = 0.15
    MOMENTUM_2 = 0.4
    MOMENTUM_3 = 0.8

    # Train 2 models with different momentum 0.0 and 0.5
    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_1} momentum model TRAINING')
    m1_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_1, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m1_history = m1_model.fit(x=train_x_avg, y=train_labels, epochs=EPOCHS, validation_data=(val_x_avg, val_labels))
    m1_eval_results = m1_model.evaluate(test_x_avg, test_labels)

    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_2} momentum model TRAINING')
    m2_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_2, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m2_history = m2_model.fit(x=train_x_avg, y=train_labels, epochs=EPOCHS, validation_data=(val_x_avg, val_labels))
    m2_eval_results = m2_model.evaluate(test_x_avg, test_labels)

    print(f'{LEARNING_RATE} learning rate, {MOMENTUM_3} momentum model TRAINING')
    m3_model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM_3, nesterov=False),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    m3_history = m3_model.fit(x=train_x_avg, y=train_labels, epochs=EPOCHS, validation_data=(val_x_avg, val_labels))
    m3_eval_results = m3_model.evaluate(test_x_avg, test_labels)

    # Prepare plot argument for model lr = 0.1, momentum_1
    m1_loss = m1_history.history['loss']
    m1_val_loss = m1_history.history['val_loss']
    m1_acc = m1_history.history['accuracy']
    m1_val_acc = m1_history.history['val_accuracy']

    # Prepare plot argument for model lr = 0.1, momentum_2
    m2_loss = m2_history.history['loss']
    m2_val_loss = m2_history.history['val_loss']
    m2_acc = m2_history.history['accuracy']
    m2_val_acc = m2_history.history['val_accuracy']

    # Prepare plot argument for model lr = 0.1, momentum_3
    m3_loss = m3_history.history['loss']
    m3_val_loss = m3_history.history['val_loss']
    m3_acc = m3_history.history['accuracy']
    m3_val_acc = m3_history.history['val_accuracy']

    # find the y axis limits
    acc_vals = [m1_acc, m1_val_acc, m2_acc, m2_val_acc, m3_acc, m3_val_acc]
    acc_y_axis_max = max(flatten_list(acc_vals)) + 0.1
    acc_y_axis_min = min(flatten_list(acc_vals)) - 0.1
    
    loss_vals = [m1_loss, m1_val_loss, m2_loss, m2_val_loss, m3_loss, m3_val_loss]
    loss_y_axis_max = max(flatten_list(loss_vals)) + 0.1
    loss_y_axis_min = min(flatten_list(loss_vals)) - 0.1


    epochs_range = range(EPOCHS)

    # Plot the 3 different learning rates
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, m1_acc, label='Training')
    plt.plot(epochs_range, m1_val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_1}')

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, m2_acc, label='Training Accuracy')
    plt.plot(epochs_range, m2_val_acc, label='Validation Accuracy')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_2}')

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, m3_acc, label='Training Accuracy')
    plt.plot(epochs_range, m3_val_acc, label='Validation Accuracy')
    plt.ylim(acc_y_axis_min, acc_y_axis_max)
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate = {LEARNING_RATE}\n Momentum = {MOMENTUM_3}')


    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, m1_loss, label='Training Loss')
    plt.plot(epochs_range, m1_val_loss, label='Validation Loss')
    plt.ylabel(f'Loss', fontsize=12)
    plt.xlabel(f"\nTest ds loss: {int(round(m1_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m1_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, m2_loss, label='Training Loss')
    plt.plot(epochs_range, m2_val_loss, label='Validation Loss')
    plt.xlabel(f"\nTest ds loss: {int(round(m2_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m2_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, m3_loss, label='Training Loss')
    plt.plot(epochs_range, m3_val_loss, label='Validation Loss')
    plt.xlabel(f"\nTest ds loss: {int(round(m3_eval_results[0], 2)*100)}%\nTest ds accuracy: {int(round(m3_eval_results[1], 2)*100)}%")
    plt.ylim(loss_y_axis_min, loss_y_axis_max)

    plt.show()

if __name__ == '__main__':

    import_model = task_2()
    # import_model.summary()
    # flower_model = task_3(import_model)
    # flower_model.summary()

    # train_ds, val_ds, test_ds = task_4()
    # history = task_5(flower_model, train_ds, val_ds)
    # task_6(history)
    # task_7(import_model, train_ds, val_ds)
    # task_8(import_model, train_ds, val_ds, test_ds)

    train_x_avg, train_labels, val_x_avg, val_labels, test_x_avg, test_labels = task_9()
    task_10(train_x_avg, train_labels, val_x_avg, val_labels, test_x_avg, test_labels)

