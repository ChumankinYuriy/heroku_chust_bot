import asyncio
import random
from datetime import datetime
from core import load_square_image, VggFeaturesWithStyleTransferLosses, style_transfer, unloader


def main():
    start_time = datetime.now()

    content_img = load_square_image('styles/4.jpg')
    style_img = load_square_image('styles/3.jpg')
    res_filename = 'tmp/' + str(random.randint(0, 999999)) + '.png'
    model = VggFeaturesWithStyleTransferLosses(content_img, style_img)
    output = style_transfer(model, content_img.clone(), 40, 1, 1e+6)
    unloader(output.squeeze(0)).save(res_filename)

    print(datetime.now() - start_time)


if __name__ == '__main__':
    main()

