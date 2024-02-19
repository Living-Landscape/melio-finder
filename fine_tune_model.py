#  fine-tuning

import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import warnings

warnings.filterwarnings('ignore')

for i in range(1, 6):
    # Load trained model
    model = keras.models.load_model("model_600_%s.keras" % i)

    # Unlock last layers in conv_base for training
    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-4]:
        layer.trainable = False

    # Create datasets
    train_dir = '%s_g/training' % i
    val_dir = '%s_g/validation' % i
    classes = os.listdir(train_dir)
    print(train_dir, val_dir)

    train_dataset = ImageDataGenerator(validation_split=0, horizontal_flip=True, vertical_flip=True,
                                       preprocessing_function=keras.applications.vgg16.preprocess_input)
    val_dataset = ImageDataGenerator(validation_split=0.99999999,
                                     preprocessing_function=keras.applications.vgg16.preprocess_input)

    train_datagen = train_dataset.flow_from_directory(train_dir, target_size=(600, 600), batch_size=16,
                                                      class_mode='binary', subset='training')
    val_datagen = val_dataset.flow_from_directory(val_dir, target_size=(600, 600), batch_size=16,
                                                  class_mode='binary', subset='validation')

    # model definition
    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="fine_tuning_600_%s.keras" % i,
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    # fine-tune neural network
    history = model.fit(train_datagen, epochs=30, validation_data=val_datagen, callbacks=callbacks)
