# use a trained CNN to predict melioration visibility on images

import keras
from keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('ignore')

# load model
model = keras.models.load_model("model_600_1.keras")

# predict melioration visibility on all images
test_dir = "images"

test_dataset = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)
test_datagen = test_dataset.flow_from_directory(test_dir, target_size=(600, 600), batch_size=16, class_mode=None,
                                                shuffle=False)
predictions = model.predict_generator(test_datagen)

print("Found meliorations on the following images:")
for i in range(0, len(predictions)):
    if predictions[i] > 0.5:
        print(test_datagen.filenames[i])
