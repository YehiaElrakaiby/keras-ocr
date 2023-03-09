import pathlib
import string
import itertools

path = pathlib.Path(__file__).parent
config_path = path.joinpath('config')
words_list_file_path = path.joinpath(f'{config_path}/words_list.txt')
# print(f'words_list_file_path is {words_list_file_path}')

characters_set = set()

with open(words_list_file_path, encoding="utf-8") as f:
    words = f.readlines()
    words_list = [s.replace("\n", "") for s in words]
    for word in words:
        for char in word:
            if char != '\n':
                characters_set.add(char)
    characters = list(characters_set)
    characters.sort()

alphabet = ''.join(characters) + string.digits
# print(f'alphabet is {alphabet}')

# Generate all combinations of length 3
iter_sentences_3_words = list(itertools.combinations(words_list, 3))
iter_sentences_2_words = list(itertools.combinations(words_list, 2))

sentences_3_words = [f'{a[0]} {a[1]} {a[2]}' for a in iter_sentences_3_words]
sentences_2_words = [f'{a[0]} {a[1]}' for a in iter_sentences_2_words]




