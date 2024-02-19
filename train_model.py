# convolutional neural network for image classification - contains or does not contain meliorations

import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import warnings

warnings.filterwarnings('ignore')

# Use VGG-16 convolutional base
conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False
)
conv_base.trainable = False

for i in range(1, 6):
    # Create datasets
    train_dir = '%s_g/training' % i
    val_dir = '%s_g/validation' % i
    classes = os.listdir(train_dir)
    print(train_dir, val_dir)

    train_dataset = ImageDataGenerator(validation_split=0, horizontal_flip=True, vertical_flip=True,
                                       preprocessing_function=keras.applications.vgg16.preprocess_input)
    val_dataset = ImageDataGenerator(validation_split=0.99999999,
                                     preprocessing_function=keras.applications.vgg16.preprocess_input)

    train_datagen = train_dataset.flow_from_directory(train_dir, target_size=(600, 600), batch_size=32,
                                                      class_mode='binary', subset='training')
    val_datagen = val_dataset.flow_from_directory(val_dir, target_size=(600, 600), batch_size=32,
                                                  class_mode='binary', subset='validation')

    # Model definition
    inputs = keras.Input(shape=(600, 600, 3))
    x = conv_base(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="model_600_%s.keras" % i,
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    # Train neural network
    history = model.fit(train_datagen, epochs=30, validation_data=val_datagen, callbacks=callbacks)
