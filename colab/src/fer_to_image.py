# -*- coding: utf-8 -*-
import csv
import os
from PIL import Image
import numpy as np

datasets_path = r'./datasets'
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')

train_set = os.path.join(datasets_path, 'train')
val_set = os.path.join(datasets_path, 'val')
test_set = os.path.join(datasets_path, 'test')
def fer_to_image():
    for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num = 1
        with open(csv_file) as f:
            csvr = csv.reader(f)
            header = next(csvr)
            for i, (label, pixel) in enumerate(csvr):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                im = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                print(image_name)
                im.save(image_name)


if __name__ == "__main__":
    fer_to_image()