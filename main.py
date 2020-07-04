import logging
import os

from aiogram import Bot, types, md
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from core import core
from utils import BotStates, parse_style_id, parse_image_type, default_styles, get_photo, ImageTypes, download_file

# Url для загрузки предобученной сети выделения признаков.
PRETRAINED_URL = 'https://drive.google.com/u/0/uc?id=1l7Lyy9a_nC9ngyCgHwy_Ex9LtO3FA4Bh&export=download'
# Имя файла в котором хранится предобученная сеть.
PRETRAINED_FILENAME = 'style_transfer.cnn'
# Токен подключения к боту.
TOKEN = os.environ['TOKEN']
# Относительный url приложения.
WEBHOOK_PATH = '/webhook/'
# Фильтр с которого принимаются запросы.
WEBAPP_HOST = '0.0.0.0'
# Порт на котором следует принимать запросы.
WEBAPP_PORT = os.environ.get('PORT')
# WEBHOOK_HOST должен содержать адрес на который будут направляться оповещения.
# например: 'https://deploy-chust-bot.herokuapp.com'
WEBHOOK_HOST = os.environ.get('WEBHOOK_HOST')
# Абсолютный url приложения.
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Конфигурация Приложения.
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())
logging.basicConfig(level=logging.DEBUG)

# Строка справки.
help_str = \
        "При работе со мной используйте команды:\n" + \
        "* '/Справка' - для просмотра справки\n" + \
        "* '/Фото' - для задания фото на которое будет перенесён стиль\n" + \
        "* '/Стиль' - для задания стиля\n" + \
        "* '/Результат' - для выполнения переноса стиля." + \
        " Перед выполнением переноса стиля необходимо задать /Фото и /Стиль\n" + \
        "* '/Покажи' - для просмотра изображения, через пробел необходимо указать, что именно вы хотите увидеть." + \
        " Можно просматривать фото, стиль, изображения стандартных стилей \n(пример: '/Покажи фото')\n" + \
        " Для просмотра стандартного стиля укажите его номер\n(пример: '/Покажи 1') \n"


@dp.message_handler(content_types=[types.ContentType.ANY], state=BotStates.PROCESSING)
async def processing(message: types.Message, state: FSMContext):
    await message.answer('Подождите, сначала я должен обработать изображения.')


@dp.message_handler(commands='start', state='*')
async def start_handler(message: types.Message, state: FSMContext):
    await bot.send_message(
        message.chat.id,
        f'Приветствую! Это демонтрационный бот для переноса стиля\n' + help_str +
        f'Подробная информация на '
        f'{md.hlink("github", "https://github.com/ChumankinYuriy/heroku_chust_bot")}',
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True)


@dp.message_handler(commands='справка', state='*')
async def help_handler(message: types.Message, state: FSMContext):
    await bot.send_message(
        message.chat.id,
        help_str +
        f'Подробная информация на '
        f'{md.hlink("github", "https://github.com/ChumankinYuriy/heroku_chust_bot")}',
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True)


@dp.message_handler(commands='покажи', state='*')
async def show_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    style_id = parse_style_id(message.text)
    image_type = parse_image_type(message.text)
    if (style_id is None) and (image_type is None):
        await message.answer('Не могу понять, что именно вы просите показать.')
        return
    photo = None
    caption = None
    if style_id is not None:
        caption = default_styles[style_id]['name']
        photo = get_photo(style_id)
    elif image_type == ImageTypes.STYLE:
        if 'style_file_id' not in user_data:
            await message.answer('Сначала задайте стиль командой \'/Стиль\'.')
            return
        caption = 'Выбранный стиль'
        photo = get_photo(user_data['style_file_id'])
    elif image_type == ImageTypes.CONTENT:
        if 'content_file_id' not in user_data:
            await message.answer('Сначала задайте фото командой \'/Фото\'.')
            return
        caption = 'Выбранное фото'
        photo = get_photo(user_data['content_file_id'])
    elif image_type == ImageTypes.RESULT:
        await message.answer('Для получения результата воспользуйтесь командой \'/Результат\'.')
        return
    await message.answer_photo(photo, caption=caption)


@dp.message_handler(commands='стиль', state='*')
async def set_style_handler(message: types.Message, state: FSMContext):
    await BotStates.WAIT_STYLE.set()
    styles = ''
    for style in default_styles.items():
        styles += str(style[0]) + '. ' + str(style[1]['name']) + '\n'
    await bot.send_message(
        message.chat.id,
        f'Я могу оформить ваше фото в таком стиле:\n' + styles +
        f'\nЧтобы выбрать стиль напишите его номер. '
        f'Чтобы посмотреть стиль напишите команду \'/Покажи\' перед номером.\n'
        f'Если хотите использовать свою картинку, то прикрепите её к следующему сообщению вместо номера.')


@dp.message_handler(state=BotStates.WAIT_STYLE)
async def set_style(message: types.Message, state: FSMContext):
    style_id = parse_style_id(message.text)
    if style_id is None:
        await message.answer('Не могу понять, какой стиль вы хотите задать.')
        return
    await state.update_data(style_file_id=style_id)
    await BotStates.DEFAULT.set()
    await message.answer('Задан стиль \'' + default_styles[style_id]['name'] + '\'.')


@dp.message_handler(content_types=[types.ContentType.PHOTO], state=BotStates.WAIT_STYLE)
async def style_photo_handler(message: types.Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    await state.update_data(style_file_id=file.file_id)
    await BotStates.DEFAULT.set()
    await message.answer('Задан новый стиль')


@dp.message_handler(commands='фото', state='*')
async def set_content_handler(message: types.Message, state: FSMContext):
    await BotStates.WAIT_CONTENT.set()
    await message.answer(f'К следующему сообщению прикрепите фото на которое хотите перенести стиль.')


@dp.message_handler(content_types=[types.ContentType.PHOTO], state=BotStates.WAIT_CONTENT)
async def content_photo_handler(message: types.Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    await state.update_data(content_file_id=file.file_id)
    await BotStates.DEFAULT.set()
    await message.answer('Задано новое фото для переноса стиля')


@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def random_photo_handler(message: types.Message):
    await bot.send_message(message.chat.id, 'Принял фотографию, но не понял зачем.')


@dp.message_handler(commands='результат', state='*')
@dp.async_task
async def get_result_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    if 'content_file_id' not in user_data:
        await message.answer("Сначала задайте фото командой '/Фото'")
        return
    if 'style_file_id' not in user_data:
        await message.answer("Сначала задайте стиль командой '/Стиль'")
        return
    await BotStates.PROCESSING.set()
    content_file = await bot.get_file(user_data['content_file_id'])
    content_filename = 'tmp/' + user_data['content_file_id'] + '.png'
    await content_file.download(content_filename)
    style_filename = None
    if user_data['style_file_id'] in default_styles:
        style_filename = default_styles[user_data['style_file_id']]['file']
    else:
        style_file = await bot.get_file(user_data['style_file_id'])
        style_filename = 'tmp/' + user_data['style_file_id'] + '.png'
        await style_file.download(style_filename)
    await message.answer(
        'Обрабатываю изображения, это может занять несколько минут. Пришлю результат как только всё будет готово.')
    result_filename = await core(content_filename, style_filename, PRETRAINED_FILENAME)
    os.remove(content_filename)
    if user_data['style_file_id'] not in default_styles: os.remove(style_filename)
    await BotStates.DEFAULT.set()
    await message.answer_photo(open(result_filename, 'rb'))
    os.remove(result_filename)


@dp.message_handler(content_types=[types.ContentType.ANY], state='*')
async def random_handler(message: types.Message, state: FSMContext):
    await bot.send_message(message.chat.id,
                           "Для общения со мной используйте команды. Просмотреть список команд: '/Справка'")


async def on_startup(dp):
    if not os.path.isfile(PRETRAINED_FILENAME):
        download_file(PRETRAINED_URL, PRETRAINED_FILENAME)
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    pass


if __name__ == '__main__':
    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH,
                  on_startup=on_startup, on_shutdown=on_shutdown,
                  host=WEBAPP_HOST, port=WEBAPP_PORT)
