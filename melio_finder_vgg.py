# use a trained CNN to predict melioration visibility on images

import os
import keras
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import matplotlib.pyplot as plt
import numpy as np

# load model
model = keras.models.load_model("melio_finder_vgg.keras")

# parameters
img_size = (512, 512)
batch_size = 8

# load images
img_dir = "images"

img_paths = sorted(
    [
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ]
)

def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
):

    def load_img_masks(input_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")
        return input_img

    dataset = tf_data.Dataset.from_tensor_slices(input_img_paths)
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


# visualise predicted masks

dataset = get_dataset(
    batch_size, img_size, img_paths
)
preds = model.predict(dataset)


def display_mask(mask, i):
    mask *= 255
    mask = np.where(mask > 127.5, 255, 0)
    plt.axis("off")
    plt.imshow(mask)
    plt.colorbar()
    plt.savefig("pred_%s" % i)
    plt.clf()


for k in range(0, len(preds)):
    display_mask(preds[k], k)
