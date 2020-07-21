import os
from matplotlib import pyplot as plt
emo_labels = ['neutral', 'happiness', 'surprise', 'sad', 'angry', 'disgust', 'fear']


# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))


if __name__ == '__main__':
    base_dir = '/Applications/PycharmProject/My_thesis/My_FER_System/GoogLeNet/FERplus_enhancement/test/'
    List = []

    for n in range(7):
        path = base_dir+str(n)
        print(path)
        ls = os.listdir(path)
        count = 0
        for i in ls:
            if os.path.isfile(os.path.join(path, i)):
                count += 1
        List.append(count)
    print(List)


    autolabel(plt.bar(range(len(List)), List, color='rgb', tick_label=emo_labels))
    plt.show()

    plt.axes(aspect=1)
    plt.pie(x=List, labels=emo_labels, autopct='%3.1f %%')
    plt.show()
