from enum import Enum
import re

from aiogram.dispatcher.filters.state import StatesGroup, State
from pip._vendor import requests

# Url для загрузки предобученной сети выделения признаков.
PRETRAINED_URL = 'https://drive.google.com/u/0/uc?id=1fAHu8nHH6c0ykQZd0EaFHnRhZENfAbh2&export=download'
# Имя файла в котором хранится предобученная сеть.
PRETRAINED_FILENAME = 'style_transfer.cnn'
# Стандартные стили.
default_styles = {
    1: {'name': 'Кубизм', 'file': 'styles/1.jpg'},
    2: {'name': 'Импрессионизм', 'file': 'styles/2.jpg'},
    3: {'name': 'Постимпрессионизм', 'file': 'styles/3.jpg'},
    4: {'name': 'Акварель', 'file': 'styles/4.jpg'},
    5: {'name': 'Хохлома', 'file': 'styles/5.jpg'},
    6: {'name': 'Гжель', 'file': 'styles/6.jpg'},
    7: {'name': 'Золотые узоры', 'file': 'styles/7.jpg'},
    8: {'name': 'Листва', 'file': 'styles/8.jpg'},
    9: {'name': 'Кора', 'file': 'styles/9.jpg'},
    10: {'name': 'Лёд', 'file': 'styles/10.jpg'},
}


# Состояния бота.
class BotStates(StatesGroup):
    DEFAULT = State()       # Свободен.
    WAIT_STYLE = State()    # Ожидает задание стиля.
    WAIT_CONTENT = State()  # Ожидает задания содержания.
    WAIT_FEEDBACK = State()  # Ожидает отзыва пользователя.
    PROCESSING = State()    # Обрабатывает изображения.


# Типы изображений.
class ImageTypes(Enum):
    CONTENT = 0  # Содержание.
    STYLE = 1    # Стиль.
    RESULT = 2   # Результат переноса стиля.


def get_photo(photo_id):
    """
    Получить фото.
    :param photo_id: str
        Если в качестве id передан telegram id фото, то он же и возвращается.
        Если в качестве id передан id стандартного стиля (см. default_styles),
         то возвращается бинарное содержание изображения стиля.
    :return: str
        id или бинарные данные.
    """
    if photo_id in default_styles:
        filename = default_styles[photo_id]['file']
        photo = open(filename, 'rb')
    else:
        photo = photo_id
    return photo


def parse_style_id(text):
    """
    Получить id стандартного стиля (см. default_styles) из текста.
    :param text: str
        Текст.
    :return: int|None
        id стиля, если id найден в тексте.
        None если id не найден в тексте.
    """
    match = re.match('.*([1-9][0-9]*).*', text.lower())
    if (match is not None) and (int(match.group(1)) in default_styles):
        return int(match.group(1))


def parse_image_type(text):
    """
    Получить тип изображения (см. ImageTypes) из текста.
    :param text: str
        Текст.
    :return: ImageTypes
        Тип изображения.
    """
    match = re.match('.*(фото|стиль|результат).*', text.lower())
    if match is None: return None
    type = match.group(1)
    if type == 'фото':
        return ImageTypes.CONTENT
    elif type == 'стиль':
        return ImageTypes.STYLE
    elif type == 'результат':
        return ImageTypes.RESULT


def download_file(url, dst):
    """
    Скачать файл по url.
    :param url: str
        url.
    :param dst: str
        имя файла для сохранения.
    """
    r = requests.get(url, allow_redirects=True)
    open(dst, 'wb').write(r.content)