import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from IPython.display import clear_output
import random

#Размер к которому будут отмасштабированы картинки.
imsize = 128
# Набор преобразований перед входом на сеть (изменение размера и преобразование из картинки в тензор).
preprocessor = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
    ])
# Преобразование из тензора в картинку.
unloader = transforms.ToPILImage()
vgg_file = 'vgg.cnn'

#cnn = None
# Загрузка предобученной vgg
#if os.path.isfile(vgg_file):
#    cnn = torch.load(vgg_file)
#else:
#    cnn = models.vgg19(pretrained=True).features.eval()
#    torch.save(cnn, vgg_file)
#cnn.eval()

def load_square_image(path):
    """
    Загрузить картинку и обрезать её до квадратного размера.
    :param path: str
        Путь, имя файла
    :return: Image
        Загруженная и обрезанная картинка.
    """
    image = Image.open(path)
    image = transforms.CenterCrop(min(image.width, image.height))(image)
    image = preprocessor(image).unsqueeze(0)
    return image


def core(content_path: str, style_path: str, tmp_dir='tmp/'):
    """
    Выполнить перенос стиля.
    :param content_path: str
        Путь до файла контента.
    :param style_path: str
        Путь до файла стиля.
    :param tmp_dir: str
        Папка для хранения временного файла результата.
    :return: Путь и имя файла с результатом переноса стиля.
    """
    content = load_square_image(content_path)
    res_filename = tmp_dir + str(random.randint(0,0,999999)) + '.png'
    unloader(content.squeeze(0)).save(res_filename)
    return res_filename


# Класс функции потерь содержания для переноса стиля.
class ContentLoss(nn.Module):

    def __init__(self, target):
        """
        Конструктор
        :param target: FloatTensor
             Карта признаков на которую должно быть похоже содержание.
        """
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# Класс функции потерь стиля для переноса стиля.
class StyleLoss(nn.Module):

    def gram_matrix(self, input):
        """
        Вычислить матрицу Грама.
        :param input: FloatTensor
            Изображение для которого вычисляется матрица.
        :return: FloatTensor
            Матрица Грама.
        """
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def __init__(self, target_feature):
        """
        Конструктор.
        :param target_feature: FloatTesor
             Карта признаков на которую должен быть стиль.
        """
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        """
        Выполнить расчёт выхода и сохранить значение функции потерь в поле loss.
        :param input: FloatTensor
            Вход слоя.
        :return: FloatTensor
            Выход слоя, равен входу.
        """
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# Класс для приведения статистик входного изображения к заданным значениям.
class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]),
                 std=torch.tensor([0.229, 0.224, 0.225])):
        """
        Конструктор.
        :param mean: FloatTensor
            Тензор средних значений по каналам (каналов должно быть 3).
        :param std: FloatTensor
            Тензор стандартных отклонений по каналам (каналов должно быть 3).
        """
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = mean.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        """
        Приветси статистики к фиксированным значениям.
        :param img: FloatTensor
            Тензор изображения над которым выполняется преобразование.
        :return: FloatTensor
            Преобразованный тензор.
        """
        return (img - self.mean) / self.std


# Модель для получения сети, которая рассчитывает функции потерь для переноса стиля.
class VggFeaturesWithStyleTransferLosses(nn.Module):
    def __init__(self, content_img, style_img):
        """
        Конструктор.
        :param content_img: FloatTensor
            Содержание.
        :param style_img: FloatTensor
            Стиль.
        """
        super(VggFeaturesWithStyleTransferLosses, self).__init__()
        cnn = models.vgg11(pretrained=True).features.eval()
        self.layers = nn.Sequential(Normalization())
        self.content_loss_layers = []
        self.style_loss_layers = []
        i = 0
        for layer in cnn.children():
            # Копирование в сеть слоёв из vgg.
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError(
                    'Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.layers.add_module(name, layer)

            # Если текущий слой свёрточный, то после него надо добавить loss слои.
            if isinstance(layer, nn.Conv2d):
                # Чтобы ускорить вычисления, будем добавлять расчёт функций потерь не после каждого свёрточного слоя.
                if i==1:
                    target = self.layers(content_img).detach()
                    content_loss = ContentLoss(target)
                    self.layers.add_module("content_loss_{}".format(i), content_loss)
                    self.content_loss_layers.append(content_loss)
                if True:
                    target_feature = self.layers(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    self.layers.add_module("style_loss_{}".format(i), style_loss)
                    self.style_loss_layers.append(style_loss)

        for i in range(len(self.layers) - 1, -1, -1):
            if isinstance(self.layers[i], ContentLoss) or isinstance(self.layers[i], StyleLoss):
                break
        self.layers = self.layers[:(i + 1)]

    def forward(self, input):
        """
        Выполнить прямое распространение по сети
        с сохранением значений функций потерь в поля content_loss и style_loss.
        :param input: FloatTensor
            Вход сети.
        :return: FloatTensor
            Выход сети.
        """
        output = self.layers(input)
        self.style_loss = 0
        for sl in self.style_loss_layers:
          self.style_loss += sl.loss
        self.content_loss = 0
        for cl in self.content_loss_layers:
          self.content_loss += cl.loss
        return output


def style_transfer(model, input_img, num_steps=300,
                   content_weight=1, style_weight=1000000, std_out=True):
    """
    Выполнить перенос стиля.
    :param model: VggFeaturesWithStyleTransferLosses
        Модель, осуществляющая вычисление значений функций потерь.
    :param input_img: FloatTensor
        Буфер для сохранения результата с начальным приближением.
    :param num_steps: int
        Количество итераций, которое будет выполнено.
    :param content_weight: float
        Вес функции потерь содержания в суммарных потерях.
    :param style_weight: float
        Вес функции потерь стиля в суммарных потерях.
    :param std_out: Boolean
        Выводить ли лог в стандартный поток вывода.
    :return: FloatTensor
        Результат переноса стиля.
    """
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    print('Optimizing..')
    cur_step = [0]

    def closure():
        # correct the values of updated input image
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        loss = content_weight * model.content_loss + style_weight * model.style_loss
        loss.backward()
        if std_out:
            print("run {}:".format(cur_step))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                model.style_loss.item(), model.content_loss.item()))
        cur_step[0] += 1
        return loss

    while cur_step[0] < num_steps:
        optimizer.step(closure)
    # a last correction...
    input_img.data.clamp_(0, 1)
    return input_img


