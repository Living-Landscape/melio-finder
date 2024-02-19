# test trained model

import keras
from keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('ignore')

# load model
model = keras.models.load_model("model_600_1.keras")

# load data
test_dir = "test_dir"  # should contain two directories - /N and /Y

test_dataset = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)
test_datagen = test_dataset.flow_from_directory(test_dir, target_size=(600, 600), batch_size=32, class_mode='binary')

# test model
model.evaluate(test_datagen)
