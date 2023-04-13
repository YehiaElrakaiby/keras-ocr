import itertools
import math
import os
import pathlib
import string
from datetime import datetime

from matplotlib import pyplot as plt

import keras_ocr
import tensorflow as tf
import sklearn
import sklearn.model_selection
import essential_generators


class KerasOCRCustomizer:
    @property
    def alphabet(self):
        return self.__alphabet

    @property
    def fonts(self):
        return self.__fonts

    @property
    def backgrounds(self):
        return self.__backgrounds

    @property
    def background_splits(self):
        return self.__background_splits

    def __init__(self):
        self.data_dir = './resources'
        self.__fonts = []
        self.__backgrounds = []
        self.__words_list = []
        self.__alphabet = ''
        self.__sentences_3_words = []
        self.__sentences_2_words = []
        self.__background_splits = ()

        self.__construct_alphabet()
        self.__recognizer_alphabet = ''.join(sorted(set(self.__alphabet.lower())))

    def download_fonts(self):
        self.__fonts = keras_ocr.data_generation.get_fonts(
            alphabet=self.alphabet,
            cache_dir=self.data_dir
        )
        return self.__fonts

    def download_backgrounds(self):
        self.__backgrounds = keras_ocr.data_generation.get_backgrounds(
            cache_dir=self.data_dir
        )
        return self.__backgrounds

    def __construct_alphabet(self):
        path = pathlib.Path(__file__).parent
        config_path = path.joinpath('config')
        words_list_file_path = path.joinpath(f'{config_path}/words_list.txt')
        # print(f'words_list_file_path is {words_list_file_path}')

        characters_set = set()

        with open(words_list_file_path, encoding="utf-8") as f:
            words = f.readlines()
            self.__words_list = [s.replace("\n", "") for s in words]
            for word in words:
                for char in word:
                    if char != '\n':
                        characters_set.add(char)
            characters = list(characters_set)
            characters.sort()

        self.__alphabet = ''.join(characters) + string.digits
        # print(f'alphabet is {alphabet}')

        # Generate all combinations of length 3
        iter_sentences_3_words = list(itertools.combinations(self.__words_list, 3))
        iter_sentences_2_words = list(itertools.combinations(self.__words_list, 2))

        self.__sentences_3_words = [f'{a[0]} {a[1]} {a[2]}' for a in iter_sentences_3_words]
        self.__sentences_2_words = [f'{a[0]} {a[1]}' for a in iter_sentences_2_words]

        return self.__alphabet

    def get_custom_text_generator(self, alphabet=None, lowercase=False, max_string_length=None):
        """Generates strings of sentences using only the letters in alphabet.

        Args:
            alphabet: The alphabet of permitted characters
            lowercase: Whether to convert all strings to lowercase.
            max_string_length: The maximum length of the string
        """
        gen = essential_generators.DocumentGenerator()

        gen.sentence_cache = self.__words_list \
                             + self.__sentences_2_words \
                             + self.__sentences_3_words
        while True:
            sentence = gen.sentence()
            if lowercase:
                sentence = sentence.lower()
            sentence = "".join([s for s in sentence if (alphabet is None or s in alphabet)])
            if max_string_length is not None:
                sentence = sentence[:max_string_length]
            yield sentence

    @staticmethod
    def get_train_validation_test_split(arr):
        train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
        val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
        return train, val, test

    def get_image_generators(self, fonts, backgrounds, text_generator, alphabet):
        self.__background_splits = self.get_train_validation_test_split(backgrounds)
        font_splits = self.get_train_validation_test_split(fonts)
        print(font_splits)

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
                self.__background_splits
            )
        ]
        # training_image_generator = image_generators[0]
        # validation_image_generator = image_generators[1]
        # test_image_generator = image_generators[2]

        return image_generators

    def get_customized_detector(self):
        detector = keras_ocr.detection.Detector(weights='clovaai_general')
        detector_batch_size = 1
        detector_basepath = os.path.join(self.data_dir, f'detector_{datetime.now().isoformat()}')
        detection_train_generator, detection_val_generator, detection_test_generator = [
            detector.get_batch_generator(
                image_generator=image_generator,
                batch_size=detector_batch_size
            ) for image_generator in image_generators
        ]
        detector.model.fit(
            detection_train_generator,
            steps_per_epoch=math.ceil(len(self.__background_splits[0]) / detector_batch_size),
            epochs=1000,
            workers=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
                tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
                tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
            ],
            validation_data=detection_val_generator,
            validation_steps=math.ceil(len(self.__background_splits[1]) / detector_batch_size),
            batch_size=detector_batch_size
        )
        detector.model.save(filepath=f"{self.data_dir}/models/")

        return detector

    def get_customized_recognizer(self):
        recognizer = keras_ocr.recognition.Recognizer(
            alphabet=self.__recognizer_alphabet,
            weights='kurapan'
        )
        recognizer.compile()
        for layer in recognizer.backbone.layers:
            layer.trainable = False
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
        recognizer_basepath = os.path.join(self.data_dir, f'recognizer_{datetime.now().isoformat()}')
        recognition_train_generator, recognition_validation_generator, recognition_test_generator = [
            recognizer.get_batch_generator(
                image_generator=image_generator,
                batch_size=recognition_batch_size,
                lowercase=True
            ) for image_generator in recognition_image_generators
        ]
        recognizer.training_model.fit(
            recognition_train_generator,
            epochs=1000,
            steps_per_epoch=math.ceil(len(self.__background_splits[0]) / recognition_batch_size),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=25),
                tf.keras.callbacks.CSVLogger(f'{recognizer_basepath}.csv', append=True),
                tf.keras.callbacks.ModelCheckpoint(filepath=f'{recognizer_basepath}.h5')
            ],
            validation_data=recognition_validation_generator,
            validation_steps=math.ceil(len(self.__background_splits[1]) / recognition_batch_size),
            workers=0,
            bacth_size=recognition_batch_size
        )
        return recognizer


if __name__ == '__main__':
    keras_ocr_optimizer = KerasOCRCustomizer()
    print(f"keras_ocr_optimizer.alphabet is {keras_ocr_optimizer}.alphabet")

    # Get fonts
    fonts = keras_ocr_optimizer.download_fonts()
    print(f"fonts are {fonts}")

    # Get backgrounds
    backgrounds = keras_ocr_optimizer.download_backgrounds()
    print(f"backgrounds are {backgrounds}")

    # Get Text Generator
    text_generator = keras_ocr_optimizer.get_custom_text_generator(alphabet=keras_ocr_optimizer.alphabet)
    print('The first generated text is: ', next(text_generator))

    # Get image generators
    image_generators = keras_ocr_optimizer.get_image_generators(fonts=fonts,
                                                                backgrounds=backgrounds,
                                                                text_generator=text_generator,
                                                                alphabet=keras_ocr_optimizer.alphabet)

    training_image_generator = image_generators[0]
    validation_image_generator = image_generators[1]
    test_image_generator = image_generators[2]

    # See what the first validation image looks like.
    image, lines = next(validation_image_generator)
    text = keras_ocr.data_generation.convert_lines_to_paragraph(lines)
    print(f"text of image is {lines}")
    plt.imshow(image)
    plt.show()

    assert tf.config.list_physical_devices('GPU'), 'No GPU is detected'

    # Get detector
    detector = keras_ocr_optimizer.get_customized_detector()

    # Get recognizer
    recognizer = keras_ocr_optimizer.get_customized_recognizer()

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
    plt.show()
