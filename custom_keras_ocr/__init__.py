import os
import pathlib

current_path = pathlib.Path(__file__).parent
# print(current_path)
fonts_path = current_path.joinpath('playground_steps').joinpath('fonts')
background_path = current_path.joinpath('playground_steps').joinpath('backgrounds')

font_file_names = os.listdir(fonts_path)
fonts = [f'{fonts_path}/{f}' for f in font_file_names if os.path.isdir(f'{fonts_path}/{f}')]
fonts.sort()
# print(fonts)

background_file_names = os.listdir(background_path)
backgrounds = [f'{background_path}/{f}' for f in background_file_names if f.endswith('jpg')]
backgrounds.sort()
# print(backgrounds)
