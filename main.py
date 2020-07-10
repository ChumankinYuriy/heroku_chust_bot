import logging
import os

from aiogram import Bot, types, md
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.types import ReplyKeyboardRemove, User
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from core import core
from keyboards import style_kb, init_main_keyboard, feedback_kb
from utils import BotStates, parse_style_id, parse_image_type, default_styles, get_photo, ImageTypes, download_file, \
    PRETRAINED_FILENAME, PRETRAINED_URL, CommandText, DataKeys

# Токен подключения к боту.
TOKEN = os.environ['TOKEN'] if 'TOKEN' in os.environ else None
# Аккаунт на который будет отправляться обратная связь.
FEEDBACK_CHAT_ID = os.environ['FEEDBACK_CHAT_ID'] if 'FEEDBACK_CHAT_ID' in os.environ else None
# Относительный url приложения.
WEBHOOK_PATH = '/webhook/'
# Фильтр с которого принимаются запросы.
WEBAPP_HOST = '0.0.0.0'
# Порт на котором следует принимать запросы.
WEBAPP_PORT = os.environ.get('PORT')
# WEBHOOK_HOST должен содержать адрес на который будут направляться оповещения.
# например: 'https://deploy-chust-bot.herokuapp.com'
WEBHOOK_HOST = os.environ['WEBHOOK_HOST'] if 'WEBHOOK_HOST' in os.environ else None
# Абсолютный url приложения.
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Конфигурация Приложения.
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())
logging.basicConfig(level=logging.DEBUG)

# Строки справки.
help_str = \
        "При работе со мной используйте команды:\n" + \
        "* '/справка' - для просмотра справки;\n" + \
        "* '/фото' - для задания фото на которое будет перенесён стиль;\n" + \
        "* '/стиль' - для задания стиля;\n" + \
        "* '/результат' - для выполнения переноса стиля;" + \
        "* '/покажи' - для просмотра изображения, через пробел необходимо указать, что именно вы хотите увидеть."

help_result_str = "Для переноса стиля воспользуйтесь командой '/результат'.\n" +\
                  "Во время обработки изображения будут обрезаны до квадратных."

help_print_str = "Можно просматривать фото, стиль, изображения стандартных стилей\n(пример: '/покажи фото')\n" + \
        "Для просмотра стандартного стиля укажите его номер\n(пример: '/покажи 1')."


async def run_processing(message: types.Message, user_data: dict):
    if DataKeys.CONTENT_FILE_ID not in user_data:
        await message.answer("Сначала необходимо выбрать фото.",
                             reply_markup=init_main_keyboard(user_data))
        return
    if DataKeys.STYLE_FILE_ID not in user_data:
        await message.answer("Сначала необходимо выбрать стиль.",
                             reply_markup=init_main_keyboard(user_data))
        return
    await BotStates.PROCESSING.set()
    content_file = await bot.get_file(user_data[DataKeys.CONTENT_FILE_ID])
    content_filename = 'tmp/' + user_data[DataKeys.CONTENT_FILE_ID] + '.png'
    await content_file.download(content_filename)
    style_filename = None
    if user_data[DataKeys.STYLE_FILE_ID] in default_styles:
        style_filename = default_styles[user_data[DataKeys.STYLE_FILE_ID]]['file']
    else:
        style_file = await bot.get_file(user_data[DataKeys.STYLE_FILE_ID])
        style_filename = 'tmp/' + user_data[DataKeys.STYLE_FILE_ID] + '.png'
        await style_file.download(style_filename)
    await message.answer('Обрабатываю фото, это займёт несколько минут. '
                         + 'Пришлю результат как только всё будет готово.')
    result_filename = await core(content_filename, style_filename, PRETRAINED_FILENAME)
    os.remove(content_filename)
    if user_data[DataKeys.STYLE_FILE_ID] not in default_styles:
        os.remove(style_filename)
    await message.answer_photo(open(result_filename, 'rb'),
                               'Обработка завершена. Если хотите, то следующим сообщением можете оставить отзыв 🙂',
                               reply_markup=init_main_keyboard(user_data))
    await BotStates.WAIT_FEEDBACK.set()
    os.remove(result_filename)


@dp.message_handler(content_types=[types.ContentType.ANY], state=BotStates.PROCESSING)
async def processing(message: types.Message, state: FSMContext):
    await message.answer('Подождите, сначала я должен обработать изображения.')


@dp.message_handler(state=BotStates.WAIT_FEEDBACK)
async def feedback_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await BotStates.DEFAULT.set()
    await bot.send_message(message.from_user.id, "Спасибо за обратную связь. Что вы хотите сделать дальше?",
                           reply_markup=init_main_keyboard(user_data))
    if FEEDBACK_CHAT_ID is not None:
        await bot.send_message(FEEDBACK_CHAT_ID, "#отзыв от @" + message.from_user.username + ":\n" + message.text)


@dp.message_handler(commands='start', state='*')
async def start_handler(message: types.Message, state: FSMContext):
    await BotStates.DEFAULT.set()
    await message.answer(
        f'Приветствую! Это демонтрационный бот для переноса стиля\n' + help_str +
        f'Подробная информация на '
        f'{md.hlink("github", "https://github.com/ChumankinYuriy/heroku_chust_bot")}',
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True)


@dp.message_handler(commands='справка', state='*')
async def help_handler(message: types.Message, state: FSMContext):
    await BotStates.DEFAULT.set()
    await message.answer(
        help_str +
        f'Подробная информация на '
        f'{md.hlink("github", "https://github.com/ChumankinYuriy/heroku_chust_bot")}',
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True)


@dp.message_handler(commands='покажи', state='*')
async def show_handler(message: types.Message, state: FSMContext):
    text = message.text.replace('/покажи', '')
    user_data = await state.get_data()
    style_id = parse_style_id(text)
    image_type = parse_image_type(text)
    await BotStates.DEFAULT.set()
    if not text:
        await message.answer(help_print_str)
        return
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
            await message.answer('Сначала задайте стиль командой \'/стиль\'.')
            return
        caption = 'Выбранный стиль'
        photo = get_photo(user_data['style_file_id'])
    elif image_type == ImageTypes.CONTENT:
        if 'content_file_id' not in user_data:
            await message.answer('Сначала задайте фото командой \'/фото\'.')
            return
        caption = 'Выбранное фото'
        photo = get_photo(user_data['content_file_id'])
    elif image_type == ImageTypes.RESULT:
        await message.answer('Для получения результата воспользуйтесь командой \'/результат\'.')
        return
    await message.answer_photo(photo, caption=caption)


@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def random_photo_handler(message: types.Message):
    await bot.send_message(message.chat.id, 'Принял фотографию, но не понял зачем.')


async def on_startup(dp):
    if not os.path.isfile(PRETRAINED_FILENAME):
        download_file(PRETRAINED_URL, PRETRAINED_FILENAME)
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    pass


@dp.message_handler(regexp=CommandText.SET_CONTENT + "|" + CommandText.SET_ANOTHER_CONTENT, state='*')
async def set_content_handler(message: types.Message, state: FSMContext):
    await BotStates.WAIT_CONTENT.set()
    await message.answer(f'К следующему сообщению прикрепите фото на которое хотите перенести стиль.',
                         reply_markup=None)


@dp.message_handler(content_types=[types.ContentType.PHOTO], state=BotStates.WAIT_CONTENT)
async def content_photo_handler(message: types.Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    await state.update_data(content_file_id=file.file_id)
    await BotStates.DEFAULT.set()
    user_data = await state.get_data()
    answer = 'Задано новое фото на которое будет перенесён стиль.\nЧто вы хотите сделать дальше?'
    await message.answer(answer, reply_markup=init_main_keyboard(user_data))


@dp.message_handler(regexp=CommandText.DO_TRANSFER, state='*')
async def set_content_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    if DataKeys.CONTENT_FILE_ID not in user_data:
        await message.answer("Сначала необходимо выбрать фото.", reply_markup=init_main_keyboard(user_data))
        return
    await BotStates.WAIT_STYLE.set()
    await message.answer(
        f'Выберите стиль из представленных, '
        f'либо пришлите в следующем сообщении изображение, которое хотите использовать в качестве стиля.',
        reply_markup=style_kb)


async def set_style_reply(chat_id: int, user_data: dict, answer: str):
    if DataKeys.CONTENT_FILE_ID not in user_data:
        answer += "Задайте фото на которое будет перенесён стиль командой '/фото'."
        await bot.send_message(chat_id, answer, reply_markup=init_main_keyboard(user_data))
    else:
        await bot.send_message(chat_id, answer, reply_markup=ReplyKeyboardRemove())


@dp.message_handler(state=BotStates.WAIT_STYLE)
@dp.async_task
async def set_style(message: types.Message, state: FSMContext):
    style_id = parse_style_id(message.text)
    if style_id is None:
        await message.answer('Не могу понять, какой стиль вы хотите задать.')
        return
    await state.update_data(style_file_id=style_id)
    await BotStates.DEFAULT.set()
    user_data = await state.get_data()
    answer = 'Задан стиль \'' + default_styles[style_id]['name'] + '\'.\n'
    await set_style_reply(message.from_user.id, user_data, answer)
    await run_processing(message, user_data)


@dp.message_handler(content_types=[types.ContentType.PHOTO], state=BotStates.WAIT_STYLE)
@dp.async_task
async def style_photo_handler(message: types.Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    await state.update_data(style_file_id=file.file_id)
    await BotStates.DEFAULT.set()
    user_data = await state.get_data()
    answer = 'Задан новый стиль\n'
    await set_style_reply(message.from_user.id, user_data, answer)
    await run_processing(message, user_data)


@dp.message_handler(regexp=CommandText.SHOW_STYLES, state='*')
async def show_styles_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    for style_id, style in default_styles.items():
        caption = style['name']
        photo = get_photo(style_id)
        await message.answer_photo(photo, caption=caption, reply_markup=ReplyKeyboardRemove())
    await message.answer(
        'Что вы хотите сделать дальше?',
        reply_markup=init_main_keyboard(user_data))


@dp.message_handler(content_types=[types.ContentType.ANY], state='*')
async def random_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer("Не понял команды. Что вы хотите сделать дальше?",
                         reply_markup=init_main_keyboard(user_data))


if __name__ == '__main__':
    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH,
                  on_startup=on_startup, on_shutdown=on_shutdown,
                  host=WEBAPP_HOST, port=WEBAPP_PORT)
