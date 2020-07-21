import os
import sys
import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchsummary import summary
import time

import cv2
import sys
import gc
import json
import numpy as np


# !/usr/bin/python
# -*- coding: UTF-8 -*-

import torchvision as tv
import torchvision.transforms as transforms
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ['GoogLeNet', 'googlenet']

model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

_GoogLeNetOuputs = namedtuple('GoogLeNetOuputs', ['logits', 'aux_logits2', 'aux_logits1'])


def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return GoogLeNet(**kwargs)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=7, aux_logits=True, transform_input=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)   #向上取整
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x


class Inception(nn.Module):     #Inception模块

    def __init__(self,in_channels,ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):      #辅助分支

    def __init__(self, in_channels, num_classes=7):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(x.size(0), -1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes

        return x


class BasicConv2d(nn.Module):       #Conv2d+BN+Relu

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)





def predict(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = tv.models.googlenet(pretrained=True)  # 创建一个模型
    #model = googlenet()
    #model.load_state_dict(torch.load('./params.pkl'))
    model=torch.load('./model_100.pkl',map_location=torch.device('cpu'))
    model = model.to(device)

    model.eval()  # 预测模式

    # 获取测试图片，并行相应的处理
    # img = Image.open('/content/drive/My Drive/Colab Notebooks/My_thesis/FERplusData/test/6/fer0032271.png')
    transform = transforms.Compose([transforms.Resize(48), transforms.ToTensor()])
    img = img.convert("RGB")  # 如果是标准的RGB格式，则可以不加
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        py = model(img)
    _, predicted = torch.max(py, 1)  # 获取分类结果
    classIndex_ = predicted[0]

    print('预测结果', int(classIndex_))


img_size = 48
emo_labels = ['neutral', 'happiness', 'surprise', 'sad', 'angry', 'disgust', 'fear']
num_class = len(emo_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = googlenet()
#model.load_state_dict(torch.load('./params.pkl'))
model=torch.load('./model_100_1.pkl',map_location=torch.device('cpu'))
model = model.to(device)



if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture('./emotionc.mp4')
    # cap.set(cv2.CAP_PROP_FPS, 30)
    #cap = cv2.VideoCapture(0)
    color = (0, 0, 255)  # 框住人脸的矩形边框颜色
    cascade_path = './haarcascade_frontalface_alt.xml'  # 人脸识别分类器本地存储路径，用以识别那个区域是人脸
    #cap = cv2.VideoCapture('./2.mov')
    # camera_number = 0
    # cap = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if ret == False:
            break
        # 图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(type(frame_gray))
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)
        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.1,
                                             minNeighbors=1, minSize=(120, 120))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                #images = []
                #rs_sum = np.array([0.0] * num_class)

                # 截取脸部图像提交给模型识别这是谁
                images = frame_gray[y: y + h, x: x + w]  # 这就是框出来之后的范围
                # image = cv2.resize(image, (img_size, img_size))
                # image = image * (1. / 255)
                # images.append(image)
                # images.append(cv2.flip(image, 1))
                # images.append(cv2.resize(image[2:45, :], (img_size, img_size)))

                images=Image.fromarray(np.uint8(images))

                model.eval()  # 预测模式

                # 获取测试图片，并行相应的处理
                # img = Image.open('/content/drive/My Drive/Colab Notebooks/My_thesis/FERplusData/test/6/fer0032271.png')
                transform = transforms.Compose([transforms.Resize(48), transforms.ToTensor()])
                images = images.convert("RGB")  # 如果是标准的RGB格式，则可以不加
                images= transform(images)
                images = images.unsqueeze(0)
                images = images.to(device)

                with torch.no_grad():
                    py = model(images)
                _, predicted = torch.max(py, 1)  # 获取分类结果
                classIndex_ = predicted[0]

                emo=emo_labels[int(classIndex_)]
                print('Emotion : ', emo)  # 打印出该情绪(在运行窗口打印，而不是cv窗口)

                # 绘制用来标记的方框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # 文字提示是谁（所用图,当前表情文本，文本位置，字体，字体大小，字体颜色，字的粗细）
                cv2.putText(frame, '%s' % emo, (x + 30, y + 30), font, 2, (25, 223, 255), 4)

        cv2.imshow("Welcome to my FER system !", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(1)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()