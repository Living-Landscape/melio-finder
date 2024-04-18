from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input,\
    RandomBrightness, RandomContrast, Dropout
from keras.models import Model
from keras.applications import EfficientNetB0
import os
import keras
import random
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import matplotlib.pyplot as plt
import numpy as np
# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/efficientnetb0_unet.py
# https://keras.io/examples/vision/oxford_pets_image_segmentation/

# parameters
img_size = (512, 512)
batch_size = 8

class_weight = {0: 1., 1: 10.}

# load data
input_dir = "training_images/images"
target_dir = "training_images/masks"

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png")
    ]
)

def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
):

    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size)
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        target_img = target_img/255
        return input_img, target_img

    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


def conv_block(input_c, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input_c)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input_d, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_d)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    x = Dropout(0.5)(x)
    return x


def build_effienet_unet(input_shape):
    """ Input """
    inputs = Input(shape=input_shape + (3,))
    x = RandomContrast(factor=0.2)(inputs)
    x = RandomBrightness(factor=0.2)(x)

    """ Encoder """
    encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    encoder.trainable = False

    s1 = encoder.get_layer("input_1").output                      # (512 x 512)
    s2 = encoder.get_layer("block2a_expand_activation").output    # (256 x 256)
    s3 = encoder.get_layer("block3a_expand_activation").output    # (128 x 128)
    s4 = encoder.get_layer("block4a_expand_activation").output    # (64 x 64)

    """ Bottleneck """
    b1 = encoder.get_layer("block6a_expand_activation").output    # (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                               # (64 x 64)
    d2 = decoder_block(d1, s3, 256)                               # (128 x 128)
    d3 = decoder_block(d2, s2, 128)                               # (256 x 256)
    d4 = decoder_block(d3, s1, 64)                                # (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="EfficientNetB0_UNET")
    return model


model = build_effienet_unet(img_size)
model.summary()

val_samples = 80
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# instantiate dataset for each split
train_dataset = get_dataset(
    batch_size,
    img_size,
    train_input_img_paths,
    train_target_img_paths,
)
valid_dataset = get_dataset(
    batch_size, img_size, val_input_img_paths, val_target_img_paths
)

# configure the model for training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=keras.losses.BinaryCrossentropy()
)

callbacks = [
    keras.callbacks.ModelCheckpoint("melio_finder_effnet.keras", save_best_only=True)
]

# train the model
epochs = 500
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=callbacks,
    class_weight=class_weight
)

val_dataset = get_dataset(
    batch_size, img_size, val_input_img_paths, val_target_img_paths
)
val_preds = model.predict(val_dataset)

# visualise predicted validation masks


def display_mask(mask, i):
    mask *= 255
    mask = np.where(mask > 127.5, 255, 0)
    plt.axis("off")
    plt.imshow(mask)
    plt.colorbar()
    plt.savefig("pred_%s" % i)
    plt.clf()


for k in range(0, len(val_preds)):
    display_mask(val_preds[k], k)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history.png")
