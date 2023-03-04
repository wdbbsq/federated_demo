import csv
from pylab import *
import csv

from pylab import *

mpl.rcParams['font.sans-serif'] = ['Fira Code', 'sans-serif']


def Picture(dic):
    newlist = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    names = []
    values = []
    for name, value in newlist[:10]:
        names.append(name)
        values.append(value)
    plt.bar(range(len(names)), values, tick_label=names)
    plt.show()


def PicCNNEnglish():
    index = "FullAPP"
    filename = "./backupresult/analysis-" + str(index) + ".csv"
    trainAccList = []
    trainLossList = []
    testAccList = []
    testLossList = []
    with open(filename, 'r', newline='') as dataSource:
        csvReader = csv.reader(dataSource)
        count = 0
        for row in csvReader:
            count += 1
            if count == 1:
                continue
            if len(row) < 5:
                break
            trainAccList.append(round(float(row[1]), 2))
            trainLossList.append(round(float(row[2]), 2))
            testAccList.append(round(float(row[3]), 2))
            testLossList.append(round(float(row[4]), 2))
    x = range(1, len(trainAccList) + 1)
    plt.title("CNN")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, trainAccList, label='Train', color='r', linestyle='-')
    # plt.plot(x, trainLossList, label='TrainLoss', color='b', linestyle='--')
    plt.plot(x, testAccList, label='Test', color='y', linestyle='-')
    # plt.plot(x, testLossList, label='TestLoss', color='k', linestyle='--')
    # y_ticks = np.arange(0,1,0.1)
    # plt.yticks(y_ticks)
    plt.legend()
    plt.savefig("results/CNNAcc-" + str(index) + ".png")
    # plt.show()


def PicCNN(name, title):
    filename = "../posion_DataPoisoning_FL/111.csv"
    trainAccList = []
    trainLossList = []
    testAccList = []
    testLossList = []
    with open(filename, 'r', newline='') as dataSource:
        csvReader = csv.reader(dataSource)
        count = 0
        for row in csvReader:
            count += 1
            if count == 1:
                continue
            trainAccList.append(round(float(row[1]), 3))
            # trainLossList.append(round(float(row[2]), 2))
            # testAccList.append(round(float(row[2]), 3))
            # testLossList.append(round(float(row[4]), 2))
    x = range(1, len(trainAccList) + 1)
    plt.title(title)
    # plt.title("原数据集")
    plt.xlabel("Turns")
    plt.ylabel("Accuracy")
    plt.plot(x, trainAccList, label='without defense', color='r', linestyle='-', linewidth=1, marker='o', markevery=2)
    # plt.plot(x, trainLossList, label='TrainLoss', color='b', linestyle='--')
    # plt.plot(x, testAccList, label='BTA', color='r', linestyle='--', linewidth=1, marker='o', markevery=2)
    # plt.plot(x, testLossList, label='TestLoss', color='k', linestyle='--')
    # y_ticks = np.arange(0,1,0.1)
    # plt.yticks(y_ticks)
    plt.legend()
    # plt.savefig("./zsq/" + title + ".png")
    plt.show()


def getresult():
    index = "FACGAN"
    filename = "./results/analysis-" + str(index) + ".csv"
    trainAccList = []
    trainLossList = []
    testAccList = []
    testLossList = []
    with open(filename, 'r', newline='') as dataSource:
        csvReader = csv.reader(dataSource)
        count = 0
        for row in csvReader:
            # print(len(row))
            if len(row) == 0:
                continue
            losslist = re.findall(r'-?\d+\.?\d*e?-?\d*?', row[0])
            if (len(losslist) >= 8):
                count += 1
                testAccList.append(round(float(losslist[-1]), 2))
                testLossList.append(round(float(losslist[-2]), 2))
                trainAccList.append(round(float(losslist[-3]), 2))
                trainLossList.append(round(float(losslist[-4]), 2))
        print(count)
    x = range(1, len(trainAccList) + 1)
    # font = FontProperties(fname="SimHei.ttf",size=14)
    plt.title("FACGAN")
    # plt.title("平衡数据集")
    plt.xlabel("Turns")
    plt.ylabel("Accuracy")
    plt.plot(x, trainAccList, label='Train', color='k', linestyle='-', linewidth=1, marker='^', markevery=8)
    # plt.plot(x, trainLossList, label='TrainLoss', color='b', linestyle='--')
    plt.plot(x, testAccList, label='Test', color='k', linestyle='--', linewidth=1, marker='o', markevery=8)
    # plt.plot(x, testLossList, label='TestLoss', color='k', linestyle='--')
    # y_ticks = np.arange(0,1,0.1)
    # plt.yticks(y_ticks)
    plt.legend()
    plt.savefig("results/img/CNNAcc-" + str(index) + ".png")
    # plt.show()


def getFACGAN(name, title):
    index = "ACGAN"
    filename = "./analysis-2" + ".csv"
    DLossList = []
    GLossList = []
    with open(filename, 'r', newline='') as dataSource:
        csvReader = csv.reader(dataSource)
        count = 0
        for row in csvReader:
            # print(len(row))
            if len(row) < 3:
                continue
            losslist = re.findall(r'-?\d+\.?\d*e?-?\d*?', row[0])
            DLossList.append(round(float(losslist[2]), 2))
            losslist = re.findall(r'-?\d+\.?\d*e?-?\d*?', row[1])
            GLossList.append(round(float(losslist[3]), 2))
        # print(count)
    x = range(1, len(GLossList) + 1)
    # font = FontProperties(fname="SimHei.ttf",size=14)
    plt.title(title)
    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.plot(x, DLossList, label='辨别器损失', color='r', linestyle='-')
    # plt.plot(x, trainLossList, label='TrainLoss', color='b', linestyle='--')
    plt.plot(x, GLossList, label='生成器损失', color='y', linestyle='-')
    # plt.plot(x, testLossList, label='TestLoss', color='k', linestyle='--')
    # y_ticks = np.arange(0,1,0.1)
    # plt.yticks(y_ticks)
    plt.legend()
    plt.savefig("results/" + str(index) + ".png")
    # plt.show()


def plot(filename, csv_dir, csv_title, xlabel, ylabel, legend_name, save_pic=False):
    """
    保存图片到csv文件所在的目录
    :param save_pic:
    :param filename:
    :param csv_dir:     文件目录
    :param csv_title:   图片标题
    :param xlabel:      x坐标名
    :param ylabel:      y坐标名
    :param legend_name: 图例名
    :return:
    """
    csv_file = csv_dir + filename + ".csv"
    ydata = []
    with open(csv_file, 'r', newline='') as dataSource:
        rows = csv.reader(dataSource)
        for row in rows:
            ydata.append(float(row[0]))
    xdata = range(1, len(ydata) + 1)
    plt.title(csv_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata, label=legend_name, color='r', linestyle='-')

    plt.legend()
    if save_pic:
        plt.savefig(csv_dir + filename + ".png")
    plt.show()


if __name__ == "__main__":
    # getresult()
    PicCNN("label_flip", "label_flip")
    # PicCNN("flex", "FLEX")
    # plot('111', '../poison/posion_DataPoisoning_FL/', '', 'Turns', 'Accuracy', 'label_flip')
    exit()
