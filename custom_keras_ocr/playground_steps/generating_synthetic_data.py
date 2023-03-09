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
from extract_characters import alphabet  # Alphabet is generated from words written on gym machines
from custom_keras_ocr import fonts, backgrounds

# assert tf.test.is_gpu_available(), 'No GPU is available.'

###
#
# Download fonts and background
#
###

# data_dir = '.'

###
#
# Replace alphabet with my own
#
###
# alphabet = string.digits + string.ascii_letters + '!?. '
# recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
# print(f'recognizer_alphabet is {recognizer_alphabet}')

###
#
# Get fonts directly
#
###
# Download fonts: fonts is a list of font filepaths.
# fonts = keras_ocr.data_generation.get_fonts(
#     alphabet=alphabet,
#     cache_dir=data_dir
# )
# print(f'fonts is {fonts}')

# Download backgrounds
# backgrounds = keras_ocr.data_generation.get_backgrounds(
#     cache_dir=data_dir
# )
# print(f'backgrounds is {backgrounds}')

###
#
# Generate Synthetic Data
#
###

text_generator = keras_ocr.data_generation.my_get_text_generator(alphabet=alphabet)
print('The first generated text is:', next(text_generator))


###
#
# Generate Synthetic Data
#
###
def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test


background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
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
# image, lines = next(image_generators[1])
# print(lines)
# text = keras_ocr.data_generation.convert_lines_to_paragraph(lines)
# print(text)

# print('The first generated validation image (below) contains:', text)
# plt.imshow(image)
