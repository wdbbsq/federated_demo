import csv
from typing import List

from pylab import *

config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

tick_labels_style = {
    "fontname": "Times New Roman",
    "fontsize": 12,
}


def get_data(work_dir, filename, selected_rows: List[int]):
    y_data_list = [[] for _ in range(len(selected_rows))]
    with open(work_dir + filename, 'r', newline='') as dataSource:
        rows = csv.reader(dataSource)
        next(rows)
        for row in rows:
            for i, j in enumerate(selected_rows):
                y_data_list[i].append(float(row[j]))
    return y_data_list


def get_confusion_matrix(trues, preds):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix


def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def plot(y_data: List[List[float]], legends, colors, linestyles, xlabel, ylabel, csv_title='', save_pic=False):
    if len(y_data) != len(legends) != len(colors) != len(linestyles):
        raise Exception('params length dont match')
    x_data = range(1, len(y_data[0]) + 1)
    plt.title(csv_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(**tick_labels_style)
    plt.yticks(**tick_labels_style)
    for idx, data in enumerate(y_data):
        plt.plot(x_data, data, label=legends[idx], linewidth=1, color=colors[idx], linestyle=linestyles[idx])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # backdoor baseline
    # plot(y_data=(get_data(work_dir='backdoor/logs/',
    #                       filename='CIFAR10_resnet18_4_3_ScaleTrue4_2023-03-10-14-57-58_trigger1.csv',
    #                       selected_rows=[1, 3]) +
    #              get_data(work_dir='backdoor/logs/',
    #                       filename='CIFAR10_resnet18_4_3_ScaleTrue4_2023-03-10-15-40-40_trigger1.csv',
    #                       selected_rows=[1, 3])),
    #      legends=['$\mathrm{MTA}$ 不攻击',
    #               '$\mathrm{BTA}$ 不攻击',
    #               '$\mathrm{MTA}$ 连续攻击',
    #               '$\mathrm{BTA}$ 连续攻击'],
    #      colors=['b', 'r', 'b', 'r'],
    #      linestyles=['--', '--', '-', '-'],
    #      xlabel='轮次',
    #      ylabel='准确率')

    # plot(y_data=(get_data(work_dir='backdoor/logs/',
    #                       filename='MNIST_badnets_2023-03-08-18-58-57_trigger0.csv',
    #                       selected_rows=[1]) +
    #              get_data(work_dir='backdoor/logs/',
    #                       filename='MNIST_trigger1.csv',
    #                       selected_rows=[1])
    #              ),
    #      legends=['联邦隐私保护框架',
    #               '传统联邦学习'],
    #      colors=['b', 'r'],
    #      linestyles=['-', '-'],
    #      xlabel='轮次',
    #      ylabel='准确率')

    plot(y_data=(get_data(work_dir='poison/logs/2023-03-26-12-33-07/',
                          filename='MNIST_badnets_24_20_ScaleTrue4.csv',
                          selected_rows=[2])
                 ),
         legends=[
                  'loss'],
         colors=['b', 'r'],
         linestyles=['-', '-'],
         xlabel='轮次',
         ylabel='准确率')

    pass
