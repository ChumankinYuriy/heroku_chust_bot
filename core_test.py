import asyncio
import logging
import random
from datetime import datetime

import torch

from core import load_square_image, VggFeaturesWithStyleTransferLosses, style_transfer, unloader
from utils import PRETRAINED_FILENAME


async def main():
    logging.basicConfig(level=logging.DEBUG)
    start_time = datetime.now()

    content_img = load_square_image('images/4.jpg')
    style_img = load_square_image('styles/11.jpg')
    res_filename = 'tmp/res.png'
    model = VggFeaturesWithStyleTransferLosses(content_img, style_img, PRETRAINED_FILENAME)
    #torch.save(model.layers, 'style_transfer.cnn')
    output = await style_transfer(model, content_img.clone(), 150, 0.001, 10000)
    unloader(output.squeeze(0)).save(res_filename)

    logging.debug(datetime.now() - start_time)


if __name__ == '__main__':
    asyncio.run(main())

