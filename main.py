import logging
import os
from datetime import datetime

from aiogram import Bot, types, md
from aiogram.dispatcher import Dispatcher
from aiogram.types.base import InputFile
from aiogram.utils.executor import start_webhook
import re
from enum import Enum
from aiogram.types.chat import ChatActions

TOKEN = os.environ['TOKEN']

WEBHOOK_HOST = 'https://deploy-chust-bot.herokuapp.com'  # name your app
WEBHOOK_PATH = '/webhook/'

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = os.environ.get('PORT')
IS_LOCALHOST = os.environ.get('IS_LOCALHOST')

if IS_LOCALHOST:
    WEBHOOK_HOST = 'https://compleo.serveousercontent.com'

WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
chat_descriptors = {}

default_styles = {
    1: {'name': 'Ван-Гог, Звёздная ночь', 'file': 'styles/1.jpg'},
    2: {'name': 'Клод Монэ, Маки', 'file': 'styles/2.jpg'},
    3: {'name': 'Кацусики Хокусай, Большая волна в Канагаве', 'file': 'styles/3.jpg'},
}


class States(Enum):
    DEFAULT = 0
    WAIT_STYLE = 1
    WAIT_CONTENT = 2
    PROCESSING = 3


class ImageTypes(Enum):
    CONTENT = 0
    STYLE = 1
    RESULT = 2


class ChatDescriptor:
    def __init__(self):
        self.state = States.DEFAULT
        self.style_file_id = None
        self.content_file_id = None
        self.result_file_id = None
        self.last_message_time = datetime.now()

    def heartbeat(self):
        self.last_message_time = datetime.now()


def get_descriptor(message: types.Message):
    """
    Получить описатель чата.
    :param message: types.Message
      Сообщение из чата.
    :return: ChatDescriptor
      Описатель чата.
    """
    if message.chat.id not in chat_descriptors:
        chat_descriptors[message.chat.id] = ChatDescriptor()
    chat_descriptors[message.chat.id].heartbeat()
    return chat_descriptors[message.chat.id]


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


@dp.message_handler(commands='start')
async def welcome(message: types.Message):
    desc = get_descriptor(message)
    await bot.send_message(
        message.chat.id,
        f'Приветствую! Это демонтрационный бот для переноса стиля\n'
        f'Подробная информация на '
        f'{md.hlink("github", "https://github.com/ChumankinYuriy/heroku_chust_bot")}',
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True)


@dp.message_handler(commands='стиль')
async def set_style_handler(message: types.Message):
    desc = get_descriptor(message)
    desc.state = States.WAIT_STYLE
    styles = ''
    for style in default_styles.items():
        styles += str(style[0]) + '. ' + str(style[1]['name']) + '\n'
    await bot.send_message(
        message.chat.id,
        f'Я могу оформить ваше фото в таком стиле:\n' + styles +
        f'\nЧтобы выбрать стиль напишите его номер. '
        f'Чтобы посмотреть стиль напишите команду \'\\Покажи\' перед номером.\n'
        f'Если хотите использовать свою картинку, то прикрепите её к следующему сообщению вместо номера.')


@dp.message_handler(commands='фото')
async def set_content_handler(message: types.Message):
    desc = get_descriptor(message)
    desc.state = States.WAIT_CONTENT
    await bot.send_message(
        message.chat.id,
        f'К следующему сообщению прикрепите фото на которое хотите перенести стиль.')


@dp.message_handler(commands='покажи')
async def show(message: types.Message):
    desc = get_descriptor(message)
    filename = 'styles/1.jpg'
    caption = 'Ван Гог, Звёздная ночь'
    style_id = parse_style_id(message.text)
    image_type = parse_image_type(message.text)
    if (style_id is None) and (image_type is None):
        await message.answer('Не могу понять, что именно вы просите показать.')
        return
    photo = None
    if style_id is not None:
        caption = default_styles[style_id]['name']
        photo = get_photo(style_id)
    elif image_type == ImageTypes.STYLE:
        if desc.style_file_id is None:
            await message.answer('Сначала задайте стиль командой \'/Стиль\'.')
            return
        caption = 'Выбранный стиль'
        photo = get_photo(desc.style_file_id)
    elif image_type == ImageTypes.CONTENT:
        if desc.content_file_id is None:
            await message.answer('Сначала задайте фото командой \'/Фото\'.')
            return
        caption = 'Выбранное фото'
        photo = get_photo(desc.content_file_id)
    await message.answer_photo(photo, caption=caption)


async def set_style(message: types.Message, desc: ChatDescriptor):
    style_id = parse_style_id(message.text)
    if style_id is None:
        await message.answer('Не могу понять, какой стиль вы хотите задать.')
        return
    desc.style_file_id = style_id
    desc.state = States.DEFAULT
    await message.answer('Задан стиль \'' + default_styles[style_id]['name'] + '\'.')


@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def photo_handler(message: types.Message):
    desc = get_descriptor(message)
    if desc.state == States.WAIT_STYLE:
        file = await bot.get_file(message.photo[-1].file_id)
        desc.style_file_id = file.file_id
        await message.answer('Задан новый стиль')
        return
    if desc.state == States.WAIT_CONTENT:
        file = await bot.get_file(message.photo[-1].file_id)
        desc.content_file_id = file.file_id
        await message.answer('Задано новое фото для переноса стиля')
        return
    await bot.send_message(message.chat.id, 'Принял фотографию, но не понял зачем.')


@dp.message_handler(content_types=[types.ContentType.ANY])
async def echo(message: types.Message):
    desc = get_descriptor(message)
    if desc.state == States.WAIT_STYLE:
        await set_style(message, desc)
        return

    await bot.send_message(message.chat.id, 'Для общения со мной используйте команды.')


async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    pass


async def on_shutdown(dp):
    pass


if __name__ == '__main__':
    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH,
                  on_startup=on_startup, on_shutdown=on_shutdown,
                  host=WEBAPP_HOST, port=WEBAPP_PORT)
