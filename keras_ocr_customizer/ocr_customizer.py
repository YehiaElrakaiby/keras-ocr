import pathlib
import string
import itertools
import essential_generators


class KerasOCRCustomizer:

    def __init__(self):
        path = pathlib.Path(__file__).parent
        config_path = path.joinpath('config')
        words_list_file_path = path.joinpath(f'{config_path}/words_list.txt')
        # print(f'words_list_file_path is {words_list_file_path}')

        characters_set = set()

        with open(words_list_file_path, encoding="utf-8") as f:
            words = f.readlines()
            self.words_list = [s.replace("\n", "") for s in words]
            for word in words:
                for char in word:
                    if char != '\n':
                        characters_set.add(char)
            characters = list(characters_set)
            characters.sort()

        self.alphabet = ''.join(characters) + string.digits
        # print(f'alphabet is {alphabet}')

        # Generate all combinations of length 3
        iter_sentences_3_words = list(itertools.combinations(self.words_list, 3))
        iter_sentences_2_words = list(itertools.combinations(self.words_list, 2))

        self.sentences_3_words = [f'{a[0]} {a[1]} {a[2]}' for a in iter_sentences_3_words]
        self.sentences_2_words = [f'{a[0]} {a[1]}' for a in iter_sentences_2_words]

    @staticmethod
    def custom_get_text_generator(alphabet=None, lowercase=False, max_string_length=None):
        """Generates strings of sentences using only the letters in alphabet.

        Args:
            alphabet: The alphabet of permitted characters
            lowercase: Whether to convert all strings to lowercase.
            max_string_length: The maximum length of the string
        """
        gen = essential_generators.DocumentGenerator()

        keras_customizer = KerasOCRCustomizer()
        gen.sentence_cache = keras_customizer.words_list \
                             + keras_customizer.sentences_2_words \
                             + keras_customizer.sentences_3_words
        while True:
            sentence = gen.sentence()
            if lowercase:
                sentence = sentence.lower()
            sentence = "".join([s for s in sentence if (alphabet is None or s in alphabet)])
            if max_string_length is not None:
                sentence = sentence[:max_string_length]
            yield sentence
