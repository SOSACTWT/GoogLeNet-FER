# -*- coding: utf-8 -*-
import csv
import os
#将原始数据集划分为训练集，验证集，测试集合
datasets_path = r'./datasets'
csv_file = os.path.join(datasets_path, 'fer2013.csv')
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')
def div():
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        rows = [row for row in csvr]

        trn = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
        print(len(trn))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
        print(len(val))

        tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
        print(len(tst))


if __name__ == "__main__":
    div()


