from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

from utils import default_styles, DataKeys, CommandText


# Клавиатура задания стиля.
style_kb = ReplyKeyboardMarkup(resize_keyboard=True)

# Формирование клавиатуры со стилями.
USE_CUSTOM_STYLE_KB = True
if USE_CUSTOM_STYLE_KB:
    style_kb.add(KeyboardButton(default_styles[2]['name']), KeyboardButton(default_styles[3]['name']))
    style_kb.add(KeyboardButton(default_styles[13]['name']), KeyboardButton(default_styles[4]['name']),
                 KeyboardButton(default_styles[1]['name']))
    style_kb.add(KeyboardButton(default_styles[7]['name']), KeyboardButton(default_styles[8]['name']),
                 KeyboardButton(default_styles[10]['name']))
    style_kb.add(KeyboardButton(default_styles[12]['name']), KeyboardButton(default_styles[9]['name']),
                 KeyboardButton(default_styles[11]['name']))
else:
    i = 0
    prev = None
    for style in default_styles.items():
        if i % 2 == 1:
            style_kb.add(prev, KeyboardButton(style[1]['name']))
        else:
            prev = KeyboardButton(style[1]['name'])
        i += 1
style_kb.add(KeyboardButton(CommandText.MY_STYLE))

feedback_kb = InlineKeyboardMarkup()
feedback_kb.add(InlineKeyboardButton('Да', callback_data='feedback_btn|yes'),
                InlineKeyboardButton('Нет', callback_data='feedback_btn|no'))

"""style_kb = ReplyKeyboardMarkup(resize_keyboard=True)
i = 0
prev = None
for style in default_styles.items():
    if i % 2 == 1:
        style_kb.add(prev, KeyboardButton(style[1]['name']))
    else:
        prev = InlineKeyboardButton(style[1]['name'])
    i += 1
style_kb.add(KeyboardButton(CommandText.MY_STYLE))

feedback_kb = InlineKeyboardMarkup()
feedback_kb.add(InlineKeyboardButton('Да', callback_data='feedback_btn|yes'),
                InlineKeyboardButton('Нет', callback_data='feedback_btn|no'))"""


def init_main_keyboard(user_data: dict):
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    if (DataKeys.ON_PROCESSING not in user_data) or not user_data[DataKeys.ON_PROCESSING]:
        if DataKeys.CONTENT_FILE_ID not in user_data:
            kb.add(KeyboardButton(CommandText.SET_CONTENT))
        else:
            kb.add(KeyboardButton(CommandText.DO_TRANSFER))
            kb.add(KeyboardButton(CommandText.SET_ANOTHER_CONTENT))
    kb.add(KeyboardButton(CommandText.SHOW_STYLES))
    kb.add(KeyboardButton(CommandText.README), KeyboardButton(CommandText.SHOW_RANDOM_EXAMPLE))
    return kb
