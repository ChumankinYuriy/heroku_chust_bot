import asyncio
import logging
import os
import random
from asyncio import sleep
from queue import SimpleQueue

from aiogram import Bot, types, md
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.types import ReplyKeyboardRemove
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from core import core
from keyboards import style_kb, init_main_keyboard
from utils import ImageTypes, clear_catalog, EXAMPLES_DIR, EXAMPLES_URL, EXAMPLES_ZIP, Statistics, read_examples
from utils import BotStates, parse_style_id, default_styles, download_file, \
    PRETRAINED_FILENAME, PRETRAINED_URL, CommandText, DataKeys
import zipfile

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
# Примерное время ожидания обработки одного фото, мин.
WAITING_TIME = int(os.environ['WAITING_TIME']) if 'WAITING_TIME' in os.environ else 5

# Конфигурация Приложения.
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())
logging.basicConfig(level=logging.DEBUG)

# Очередь вычислительных задач.
task_queue = SimpleQueue()
# Количество задач, которые предстоит обработать до освобождения вычислительных ресурсов.
on_processing = [0]
# Набор стандартных примеров.
examples = [{}]


async def task_queue_processing():
    """Функция обработки вычислительных задач."""
    while True:
        if not task_queue.empty():
            logging.debug('Starting processing task.')
            on_processing[0] = task_queue.qsize()
            future, task = task_queue.get()
            future.set_result(await task)
            logging.debug('Processing task was completed.')
            Statistics.process_request()
        else:
            on_processing[0] = 0
        await sleep(1)


# Строка с описанием бота.
help_str = 'Вы работаете с демонтрационным ботом для переноса стиля.\n' + \
           'Для управления используйте команды, которые отображаются ' \
           'на виртуальной клавиатуре и следуйте инструкциям в сообщениях.'


async def run_processing(message: types.Message, user_data: dict):
    if DataKeys.CONTENT_FILE_ID not in user_data:
        await message.answer("Сначала необходимо выбрать фото для обработки.",
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
    time_str = ''
    waiting_time = WAITING_TIME * (on_processing[0] + 1)
    if waiting_time == 0:
        time_str = 'менее минуты'
    elif waiting_time % 10 == 1:
        time_str = 'около ' + str(waiting_time) + ' минуты'
    else:
        time_str = 'около ' + str(waiting_time) + ' минут'
    info = 'Обрабатываю фото, это займёт ' + time_str + '. Пришлю результат как только всё будет готово.'
    await message.answer(info, reply_markup=init_main_keyboard(user_data))
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    task = core(content_filename, style_filename, PRETRAINED_FILENAME)
    task_queue.put((future,task))
    result_filename = await future
    os.remove(content_filename)
    if user_data[DataKeys.STYLE_FILE_ID] not in default_styles:
        os.remove(style_filename)
    await types.ChatActions.upload_photo()
    user_data[DataKeys.ON_PROCESSING] = False
    await message.answer_photo(open(result_filename, 'rb'),
                               'Обработка завершена. Если хотите, то следующим сообщением можете оставить отзыв 🙂',
                               reply_markup=init_main_keyboard(user_data))
    await BotStates.WAIT_FEEDBACK.set()
    os.remove(result_filename)


@dp.message_handler(regexp=CommandText.SHOW_STYLES, state='*')
async def show_styles_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    counter = 0
    for style_id, style in default_styles.items():
        # 9 картинок красиво собираются в квадрат.
        if counter == 9:
            await message.answer_media_group(media)
            media = types.MediaGroup()
            counter = 0
        media.attach_photo(types.InputFile(style['file']), caption=style['name'])
        counter += 1
    await message.answer_media_group(media)
    await message.answer(
        'Это только список стандартных стилей. '
        'Перед обработкой вы можете выбрать в качестве стиля любое понравившееся изображение в вашем телефоне.',
        reply_markup=init_main_keyboard(user_data))


@dp.message_handler(regexp=CommandText.SHOW_RANDOM_EXAMPLE, state='*')
async def show_example_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    if len(examples[0]) == 0:
        await message.answer(
            'Извиняюсь, на сервере нет доступных примеров.',
            reply_markup=init_main_keyboard(user_data))
    else:
        example = random.choice(list(examples[0].values()))
        await types.ChatActions.upload_photo()
        media = types.MediaGroup()
        media.attach_photo(types.InputFile(example[ImageTypes.RESULT]), caption='Результат')
        media.attach_photo(types.InputFile(example[ImageTypes.CONTENT]), caption='Фото')
        media.attach_photo(types.InputFile(example[ImageTypes.STYLE]), caption='Стиль')
        await message.answer_media_group(media)
        await message.answer(
            'Вот такой пример.',
            reply_markup=init_main_keyboard(user_data))


@dp.message_handler(regexp=CommandText.README, state='*')
async def readme_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer('Я могу обработать фото таким образом, '
                         'чтобы стилистически оно стало похоже на другое изображение (стиль).',
                         reply_markup=ReplyKeyboardRemove())
    await types.ChatActions.upload_photo()
    flower_id = 4
    if flower_id in examples[0]:
        media = types.MediaGroup()
        media.attach_photo(types.InputFile(examples[0][flower_id][ImageTypes.CONTENT]),
                           caption='Например, в качестве фото можно использовать цветы.')

        media.attach_photo(types.InputFile(examples[0][flower_id][ImageTypes.STYLE]),
                           caption='А в качестве стиля акварельный рисунок цветов.')
        media.attach_photo(types.InputFile(examples[0][flower_id][ImageTypes.RESULT]),
                           caption='В результате получатся цветы с исходного фото, '
                                   'но как буд-то нарисованные акварелью.')
        await message.answer_media_group(media)
    await message.answer('Обычно в качестве стиля используют '
                         'либо картинку с содержанием похожим на исходное фото, но оформленным иначе, '
                         'либо красивую текстуру.\n\n'
                         'Подробную информацию о моей реализации смотрите на '
                         f'{md.hlink("github", "https://github.com/ChumankinYuriy/heroku_chust_bot")}.',
                         reply_markup=init_main_keyboard(user_data), disable_web_page_preview=True,
                         parse_mode=types.ParseMode.HTML)


@dp.message_handler(content_types=[types.ContentType.ANY], state=BotStates.PROCESSING)
async def processing(message: types.Message, state: FSMContext):
    await message.answer('Подождите, сначала я должен обработать фото.')


@dp.message_handler(commands='start', state='*')
async def start_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await BotStates.DEFAULT.set()
    await message.answer(
        f'Приветствую!\n' + help_str,
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=init_main_keyboard(user_data)
    )


@dp.message_handler(commands='help', state='*')
async def help_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await BotStates.DEFAULT.set()
    await message.answer(
        help_str,
        parse_mode=types.ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=init_main_keyboard(user_data)
    )


@dp.message_handler(commands='statistics', state='*')
async def statistics_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await BotStates.DEFAULT.set()
    stat = Statistics.get()
    info = 'Начиная с ' + stat['date'] + ' я обработал ' + str(stat['counter'])
    mod = stat['counter'] % 10
    if mod == 1:
        info += ' изображение.'
    elif mod in [2, 3, 4]:
        info += ' изображения.'
    else:
        info += ' изображений.'
    info += '\nНа текущий момент длина очереди очереди на обработку равна ' + str(on_processing[0]) + '.'
    await message.answer(
        info,
        reply_markup=init_main_keyboard(user_data)
    )


@dp.message_handler(regexp=CommandText.SET_CONTENT + "|" + CommandText.SET_ANOTHER_CONTENT, state='*')
async def set_content_handler(message: types.Message, state: FSMContext):
    await BotStates.WAIT_CONTENT.set()
    await message.answer('К следующему сообщению прикрепите фото, которое хотите обработать.',
                         reply_markup=None)


@dp.message_handler(content_types=[types.ContentType.PHOTO], state=BotStates.WAIT_CONTENT)
async def content_photo_handler(message: types.Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    await state.update_data(content_file_id=file.file_id)
    await BotStates.DEFAULT.set()
    user_data = await state.get_data()
    answer = 'Задано новое фото для обработки.\nВо время обработки фото будет обрезано до квадратного.'
    await message.answer(answer, reply_markup=init_main_keyboard(user_data))


@dp.message_handler(regexp=CommandText.DO_TRANSFER, state='*')
async def set_content_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    if DataKeys.CONTENT_FILE_ID not in user_data:
        await message.answer("Сначала необходимо выбрать фото для обработки.",
                             reply_markup=init_main_keyboard(user_data))
        return
    await BotStates.WAIT_STYLE.set()
    await message.answer(
        'Выберите стиль из представленных.',
        reply_markup=style_kb)


async def set_style_reply(chat_id: int, user_data: dict, answer: str):
    if DataKeys.CONTENT_FILE_ID not in user_data:
        answer += "Задайте фото для обработки."
    await bot.send_message(chat_id, answer, reply_markup=init_main_keyboard(user_data))


@dp.message_handler(state=BotStates.WAIT_STYLE)
@dp.async_task
async def set_style(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    if message.text == CommandText.MY_STYLE:
        await message.answer(
            'К следующему сообщению прикрепите изображение, которое хотите использовать в качестве стиля.',
            reply_markup=ReplyKeyboardRemove())
        return

    style_id = parse_style_id(message.text)
    if style_id is None:
        await message.answer(
            'Не могу понять, какой стиль вы хотите задать.', reply_markup=init_main_keyboard(user_data))
        return
    await state.update_data(style_file_id=style_id)
    await BotStates.DEFAULT.set()
    await state.update_data(on_processing=True)
    user_data = await state.get_data()
    answer = 'Задан стиль \'' + default_styles[style_id]['name'] + '\'.\n'
    await set_style_reply(message.from_user.id, user_data, answer)
    await run_processing(message, user_data)
    await state.update_data(on_processing=False)


@dp.message_handler(content_types=[types.ContentType.PHOTO], state=BotStates.WAIT_STYLE)
@dp.async_task
async def style_photo_handler(message: types.Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    await state.update_data(style_file_id=file.file_id)
    await BotStates.DEFAULT.set()
    await state.update_data(on_processing=True)
    user_data = await state.get_data()
    answer = 'Задан новый стиль.\n'
    await set_style_reply(message.from_user.id, user_data, answer)
    await run_processing(message, user_data)
    await state.update_data(on_processing=False)


@dp.message_handler(state=BotStates.WAIT_FEEDBACK)
async def feedback_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await BotStates.DEFAULT.set()
    await bot.send_message(message.from_user.id, "Спасибо за обратную связь.",
                           reply_markup=init_main_keyboard(user_data))
    if FEEDBACK_CHAT_ID is not None:
        await bot.send_message(FEEDBACK_CHAT_ID, "#отзыв от @" + message.from_user.username + ":\n" + message.text)


@dp.message_handler(content_types=[types.ContentType.PHOTO], state='*')
async def random_photo_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await bot.send_message(message.chat.id, 'Принял фотографию, но не понял зачем.',
                           reply_markup=init_main_keyboard(user_data))


@dp.message_handler(content_types=[types.ContentType.ANY], state='*')
async def random_handler(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer("Не понял команды.",
                         reply_markup=init_main_keyboard(user_data))


def install_examples():
    logging.debug('Trying to download and unzip ' + EXAMPLES_ZIP)
    download_file(EXAMPLES_URL, EXAMPLES_ZIP)
    logging.debug(EXAMPLES_ZIP + ' was downloaded')
    with zipfile.ZipFile(EXAMPLES_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXAMPLES_DIR)
        logging.debug(EXAMPLES_ZIP + ' was unzipped')
    logging.debug(EXAMPLES_ZIP + ' is OK')


async def on_startup(dp):
    logging.debug('Startup')
    try:
        if not os.path.isfile(PRETRAINED_FILENAME):
            logging.debug('Trying to download ' + PRETRAINED_FILENAME)
            download_file(PRETRAINED_URL, PRETRAINED_FILENAME)
            logging.debug(PRETRAINED_FILENAME + ' was downloaded')
        logging.debug(PRETRAINED_FILENAME + ' is OK')
        if not os.path.exists(EXAMPLES_DIR):
            os.mkdir(EXAMPLES_DIR)
            install_examples()
        elif ('.gitignore' in os.listdir(EXAMPLES_DIR)) and (len(os.listdir(EXAMPLES_DIR)) == 1):
            install_examples()
        logging.debug('Examples are OK')
        logging.debug('Current directory is ' + os.path.abspath(os.getcwd()))
        examples[0] = read_examples()
        logging.debug('Number of examples is ' + str(len(examples[0])))
        clear_catalog('tmp/', lambda path: path != '.gitignore')
    except Exception as ex:
        logging.error('Failed while preloading: ' + str(ex))
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    pass


if __name__ == '__main__':
    dp.loop.create_task(task_queue_processing())
    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH,
                  on_startup=on_startup, on_shutdown=on_shutdown,
                  host=WEBAPP_HOST, port=WEBAPP_PORT)
