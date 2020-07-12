from datetime import datetime
from enum import Enum
import re

import logging
from aiogram.dispatcher.filters.state import StatesGroup, State
from pip._vendor import requests
import os
from os.path import isfile, join
import json

# Url для загрузки предобученной сети выделения признаков.
PRETRAINED_URL = 'https://drive.google.com/u/0/uc?id=1fAHu8nHH6c0ykQZd0EaFHnRhZENfAbh2&export=download'
# Имя файла в котором хранится предобученная сеть.
PRETRAINED_FILENAME = 'style_transfer.cnn'
# Адрес по которому расположен zip архив с примерами.
EXAMPLES_URL = 'https://drive.google.com/u/0/uc?id=1hkvqijfwePh0DcMSrP8cBdGPPLqP9yvM&export=download'
# Имя архива с примерами.
EXAMPLES_ZIP = 'examples.zip'

# Директория с примерами
EXAMPLES_DIR = 'examples/'

# Стандартные стили.
default_styles = {
    1: {'name': 'Кубизм', 'file': 'styles/1.jpg'},
    2: {'name': 'Импрессионизм', 'file': 'styles/2.jpg'},
    3: {'name': 'Постимпрессионизм', 'file': 'styles/3.jpg'},
    4: {'name': 'Акварель', 'file': 'styles/4.jpg'},
    #5: {'name': 'Хохлома', 'file': 'styles/5.jpg'},
    #6: {'name': 'Гжель', 'file': 'styles/6.jpg'},
    7: {'name': 'Золотые узоры', 'file': 'styles/7.jpg'},
    8: {'name': 'Листва', 'file': 'styles/8.jpg'},
    9: {'name': 'Кора', 'file': 'styles/9.jpg'},
    10: {'name': 'Лёд', 'file': 'styles/10.jpg'},
    11: {'name': 'Пламя', 'file': 'styles/11.jpg'},
    12: {'name': 'Геометрия', 'file': 'styles/12.jpg'},
    13: {'name': 'Космос', 'file': 'styles/13.jpg'}
}

# Клавиатура:
# 2 3
# 13 4 1
# 7 8 10
# 12 9 11


# Класс для ведения статистики.
class Statistics:
    FILENAME = 'statistics.json'

    @staticmethod
    def create_if_not_exists():
        if not os.path.exists(Statistics.FILENAME):
            with open(Statistics.FILENAME, 'w') as f:
                f.write(json.dumps({'counter': 0, 'date': datetime.now().strftime("%d.%m.%Y, %H:%M:%S")}))

    @staticmethod
    def process_request():
        """
        Учесть обработанный запрос.
        :return: None
        """
        try:
            Statistics.create_if_not_exists()
            data = None
            with open(Statistics.FILENAME, 'r') as f:
                file_content = f.read()
                data = json.loads(file_content)
            with open(Statistics.FILENAME, 'w') as f:
                data['counter'] += 1
                f.write(json.dumps(data))
        except Exception as ex:
            logging.error('Fail to account statistics: ' + str(ex))

    @staticmethod
    def get():
        """
        Получить статистику.
        :return: dict
            Статистика, ключи: counter (int) - счётчик обработанных запросов,
                               date (datetime) - дата и время начала учёта статистики.
        """
        try:
            Statistics.create_if_not_exists()
            data = None
            with open(Statistics.FILENAME, 'r') as f:
                file_content = f.read()
                data = json.loads(file_content)
            return data
        except Exception as ex:
            logging.error('Fail to read statistics: ' + str(ex))
            return {'counter': -1, 'date': datetime.now().strftime("%d.%m.%Y %H:%M:%S")}


# Типы изображений.
class ImageTypes(Enum):
    CONTENT = 0  # Содержание.
    STYLE = 1  # Стиль.
    RESULT = 2  # Результат переноса стиля.


def parse_image_type(text):
    """
    Получить тип изображения (см. ImageTypes) из текста.
    :param text: str
        Текст.
    :return: ImageTypes
        Тип изображения.
    """
    match = re.match('.*(content|style|result).*', text.lower())
    if match is None: return None
    type = match.group(1)
    if type == 'content':
        return ImageTypes.CONTENT
    elif type == 'style':
        return ImageTypes.STYLE
    elif type == 'result':
        return ImageTypes.RESULT


def read_examples():
    """
    Считать набор примеров.
    :return: dict
        Набор примеров, ключи - типы изображений (ImageTypes), значения пути до файлов.
    """
    # Заполнение массива с именами файлов примеров.
    example_files = [f for f in os.listdir(EXAMPLES_DIR) if isfile(join(EXAMPLES_DIR, f))]
    # Массив с названиями примеров.
    # {ImageTypes.CONTENT - файл содержания, ImageTypes.STYLE - файл стиля, ImageTypes.RESULT - файл результата.}
    examples = {}
    for file in example_files:
        im_type = parse_image_type(file)
        if im_type is None:
            continue
        num_match = re.findall('[1-9][0-9]*', file.lower())
        example_id = int(num_match.pop()) if (num_match is not None) and (len(num_match) != 0) else -1
        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][im_type] = EXAMPLES_DIR + file

    for example_id in list(examples.keys()):
        if (ImageTypes.RESULT not in examples[example_id]) or (ImageTypes.CONTENT not in examples[example_id]) \
                or (ImageTypes.STYLE not in examples[example_id]):
            examples.pop(example_id)
    return examples

# Состояния бота.
class BotStates(StatesGroup):
    DEFAULT = State()  # Свободен.
    WAIT_STYLE = State()  # Ожидает задание стиля.
    WAIT_CONTENT = State()  # Ожидает задания содержания.
    WAIT_FEEDBACK = State()  # Ожидает отзыва пользователя.
    PROCESSING = State()  # Обрабатывает изображения.


# Тексты команд.
class CommandText:
    SET_CONTENT = 'Запомни фото для обработки'
    SET_ANOTHER_CONTENT = 'Запомни другое фото'
    DO_TRANSFER = 'Обработай фото'
    SHOW_STYLES = 'Покажи стандартные стили'
    SHOW_RANDOM_EXAMPLE = 'Покажи случайный пример'
    README = 'Расскажи о себе'
    MY_STYLE = 'Изображение с моего телефона'


# Ключи в словаре данных чата.
class DataKeys:
    CONTENT_FILE_ID = 'content_file_id'  # id файла с содержанием.
    STYLE_FILE_ID = 'style_file_id'      # id файла со стилем.
    ON_PROCESSING = 'on_processing'      # Флаг, показывающий выполняются ли расчёты для текущего пользователя.


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
    for id, style in default_styles.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        match = re.match('^(' + style['name'].lower() + ')', text.lower())
        if match is not None:
            return id


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


def clear_catalog(src, checker):
    """
    Очистить в каталоге, всё кроме того, что пройдёт проверку лямбдой.
    :param src: str
        Зачищаемая директория.
    :param checker: lambda
        Лямбда, возвращает True если файл надо удалить, иначе False.
    :return: None
    """
    file_list = os.listdir(src)
    for item in file_list:
        s = os.path.join(src, item)
        if os.path.isdir(s):
            clear_catalog(s)
        else:
            if checker(str(os.path.basename(s))):
                try:
                    os.remove(s)
                except Exception as ex:
                    logging.error('Failed to remove file: ' + str(ex))

