import asyncio
import random
from datetime import datetime
from core import load_square_image, VggFeaturesWithStyleTransferLosses, style_transfer, unloader


async def main():
    start_time = datetime.now()

    content_img = load_square_image('test_data/2.png')
    style_img = load_square_image('styles/1.jpg')
    res_filename = 'tmp/res.png'
    model = VggFeaturesWithStyleTransferLosses(content_img, style_img)
    output = await style_transfer(model, content_img.clone(), 60, 0.001, 10000)
    unloader(output.squeeze(0)).save(res_filename)

    print(datetime.now() - start_time)


if __name__ == '__main__':
    asyncio.run(main())

