from enum import Enum
import re

from aiogram.dispatcher.filters.state import StatesGroup, State

default_styles = {
    1: {'name': 'Ван-Гог, Звёздная ночь', 'file': 'styles/1.jpg'},
    2: {'name': 'Клод Монэ, Маки', 'file': 'styles/2.jpg'},
    3: {'name': 'Кацусики Хокусай, Большая волна в Канагаве', 'file': 'styles/3.jpg'},
}


class BotStates(StatesGroup):
    DEFAULT = State()
    WAIT_STYLE = State()
    WAIT_CONTENT = State()


class ImageTypes(Enum):
    CONTENT = 0
    STYLE = 1
    RESULT = 2


def get_photo(photo_id):
    if photo_id in default_styles:
        filename = default_styles[photo_id]['file']
        photo = open(filename, 'rb')
    else:
        photo = photo_id
    return photo


def parse_style_id(text):
    match = re.match('.*([1-3]).*', text.lower())
    if (match is not None) and (int(match.group(1)) in default_styles):
        return int(match.group(1))


def parse_image_type(text):
    match = re.match('.*(фото|стиль|результат).*', text.lower())
    if match is None: return None
    type = match.group(1)
    if type == 'фото':
        return ImageTypes.CONTENT
    elif type == 'стиль':
        return ImageTypes.STYLE
    elif type == 'результат':
        return ImageTypes.RESULT
