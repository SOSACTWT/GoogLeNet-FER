from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import os
#数据增强
def data_enhance():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    dir = r'/Applications/PycharmProject/My_thesis/My_FER_System/GoogLeNet/FERPlus_data/train/0'
    newdir = r'/Applications/PycharmProject/My_thesis/My_FER_System/GoogLeNet/FERplus_data_enhancement/train/0'
    for filename in os.listdir(dir):  # listdir的参数是文件夹的路径
        print(filename)
        img = load_img(dir + '/' + filename)  # 这是一个PIL图像
        x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
        x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
        # 下面是生产图片的代码
        # 生产的所有图片保存在 `preview/` 目录下
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=newdir,
                                  save_prefix='0',
                                  save_format='jpeg'):
            i += 1
            if i > 5:
                break  # 否则生成器会退出循环

if __name__ == "__main__":
    data_enhance()