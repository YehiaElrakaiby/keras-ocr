# https://keras-ocr.readthedocs.io/en/latest/examples/end_to_end_training.html

import zipfile
import datetime
import string
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection

import keras_ocr
from keras_ocr_customizer.ocr_customizer import KerasOCRCustomizer  # Alphabet is generated from words written on gym machines
from custom_keras_ocr import fonts, backgrounds

# assert tf.test.is_gpu_available(), 'No GPU is available.'

###
#
# Download fonts and background
#
###
kers_ocr_optimizer = KerasOCRCustomizer()

data_dir = '.'

###
#
# Replace alphabet with my own
#
###
# alphabet = string.digits + string.ascii_letters + '!?. '
recognizer_alphabet = ''.join(sorted(set(kers_ocr_optimizer.alphabet.lower())))
# print(f'recognizer_alphabet is {recognizer_alphabet}')

###
#
# Get fonts directly
#
###
# Download fonts: fonts is a list of font filepaths.
fonts = keras_ocr.data_generation.get_fonts(
    alphabet=kers_ocr_optimizer.alphabet,
    cache_dir=data_dir
)
# print(f'fonts is {fonts}')

# Download backgrounds
backgrounds = keras_ocr.data_generation.get_backgrounds(
    cache_dir=data_dir
)
# print(f'backgrounds is {backgrounds}')

###
#
# Generate Synthetic Data
#
###

text_generator = kers_ocr_optimizer.custom_get_text_generator(alphabet=kers_ocr_optimizer.alphabet)
print('The first generated text is:', next(text_generator))


###
#
# Generate Synthetic Data:
#
# Data generation creates images where
# Text on images is generated by the text generator
# The font used is one of the fonts
# The background is one of the backgrounds
#
# Generators are split into train, validation, and test by separating the fonts and backgrounds used in each.
#
# Validation set: A set of examples used to tune the parameters of a classifier,
# for example to choose the number of hidden units in a neural network.
#
# Test set: A set of examples used only to assess the performance of a fully-specified classifier.
#
###
def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test


background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)
print(font_splits)

image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            kers_ocr_optimizer.alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        font_size=(60, 120),
        margin=50,
        rotationX=(-0.05, 0.05),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15)
    ) for current_fonts, current_backgrounds in zip(
        font_splits,
        background_splits
    )
]

# See what the first validation image looks like.
image, lines = next(image_generators[1])
print(lines)
text = keras_ocr.data_generation.convert_lines_to_paragraph(lines)
print(text)

# print('The first generated validation image (below) contains:', text)
# plt.imshow(image)
# plt.show()
# Save the image to a file
# plt.imsave('./my_image.png', image, cmap='rgb')

###
#
# Build Base Detector and Recognizer Models:
#
###
detector = keras_ocr.detection.Detector(weights='clovaai_general')
recognizer = keras_ocr.recognition.Recognizer(
    alphabet=recognizer_alphabet,
    weights='kurapan'
)
recognizer.compile()
for layer in recognizer.backbone.layers:
    layer.trainable = False

###
#
# Train the detector
#
# Run training until we have no improvement on the validation set for 5 epochs.
#
###
detector_batch_size = 1
detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')
detection_train_generator, detection_val_generator, detection_test_generator = [
    detector.get_batch_generator(
        image_generator=image_generator,
        batch_size=detector_batch_size
    ) for image_generator in image_generators
]
detector.model.fit(
    detection_train_generator,
    steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
    epochs=1000,
    workers=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
    ],
    validation_data=detection_val_generator,
    validation_steps=math.ceil(len(background_splits[1]) / detector_batch_size),
    batch_size=detector_batch_size
)

###
#
# Train the recognizer
#
# Note that the recognizer expects images to already be cropped to single lines of text.
# keras-ocr provides a convenience method for converting our existing generator into a single-line generator.
# So we perform that conversion.
#
###
max_length = 10
recognition_image_generators = [
    keras_ocr.data_generation.convert_image_generator_to_recognizer_input(
        image_generator=image_generator,
        max_string_length=min(recognizer.training_model.input_shape[1][1], max_length),
        target_width=recognizer.model.input_shape[2],
        target_height=recognizer.model.input_shape[1],
        margin=1
    ) for image_generator in image_generators
]

# See what the first validation image for recognition training looks like.
image, text = next(recognition_image_generators[1])
print('This image contains:', text)
plt.imshow(image)
plt.show()

###
# Just like the detector, the recognizer has a method for converting the image generator
# into a batch_generator that Keras’ fit_generator can use.
# We use the same callbacks for early stopping and logging as before.
###
recognition_batch_size = 8
recognizer_basepath = os.path.join(data_dir, f'recognizer_{datetime.datetime.now().isoformat()}')
recognition_train_generator, recognition_val_generator, recognition_test_generator = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=recognition_batch_size,
        lowercase=True
    ) for image_generator in recognition_image_generators
]
recognizer.training_model.fit(
    recognition_train_generator,
    epochs=1000,
    steps_per_epoch=math.ceil(len(background_splits[0]) / recognition_batch_size),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=25),
        tf.keras.callbacks.CSVLogger(f'{recognizer_basepath}.csv', append=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{recognizer_basepath}.h5')
    ],
    validation_data=recognition_val_generator,
    validation_steps=math.ceil(len(background_splits[1]) / recognition_batch_size),
    workers=0,
    bacth_size=recognition_batch_size
)



###
# Use the Model for Inference
###
pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)
image, lines = next(image_generators[0])
predictions = pipeline.recognize(images=[image])[0]
drawn = keras_ocr.tools.drawBoxes(
    image=image, boxes=predictions, boxes_format='predictions'
)
print(
    'Actual:', '\n'.join([' '.join([character for _, character in line]) for line in lines]),
    'Predicted:', [text for text, box in predictions]
)
plt.imshow(drawn)