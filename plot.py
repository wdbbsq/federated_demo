import csv
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix
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
    conf_matrix = confusion_matrix(trues, preds, labels=labels)
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
    # 设置图片大小
    # plt.figure(figsize=(8, 6))
    x_data = range(1, len(y_data[0]) + 1)
    plt.title(csv_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(**tick_labels_style)
    plt.yticks(**tick_labels_style)
    # x轴坐标显示为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, data in enumerate(y_data):
        plt.plot(x_data, data, label=legends[idx], linewidth=1, color=colors[idx], linestyle=linestyles[idx])

    plt.legend()
    plt.show()


def backdoor_dba(work_dir, filename):
    """
    dba基准
    """
    plot(y_data=(get_data(work_dir=work_dir,
                          filename=filename,
                          selected_rows=[1, 3])
                 ),
         csv_title='',
         legends=[
             '攻击者比例$\mathrm{ - 0\% }$',
             '攻击者比例$\mathrm{ - 20\% }$',
             '攻击者比例$\mathrm{ - 40\% }$',
         ],
         colors=['b', 'r', 'r'],
         linestyles=['-', '-', '--'],
         xlabel='轮次',
         ylabel='准确率')


def backdoor_defense():
    """
    后门防护框架性能
    """
    labels = ['$\mathrm{ MTA }$', '$\mathrm{ BTA }$']
    for i in range(2):
        ydata = get_data(work_dir='backdoor/logs/',
                         filename='dba.csv', selected_rows=[i + 1]) + \
                get_data(work_dir='backdoor/logs/',
                         filename='flex.csv',
                         selected_rows=[i + 1])
        ydata = np.asarray(ydata) * 0.01
        plot(y_data=ydata,
            csv_title='',
            legends=[
                '无防护',
                '后门消除方法',
            ],
            colors=['r', 'b'],
            linestyles=['-', '-'],
            xlabel='轮次',
            ylabel=labels[i])


def poison_baseline():
    """
    投毒攻击基准实验
    """
    plot(y_data=(get_data(work_dir='poison/logs/2023-03-26-11-18-23/',
                          filename='MNIST_badnets_10_5_ScaleTrue4.csv',
                          selected_rows=[1]) +
                 get_data(work_dir='poison/logs/2023-03-26-11-31-25/',
                          filename='MNIST_badnets_10_5_ScaleTrue4.csv',
                          selected_rows=[1]) +
                 get_data(work_dir='poison/logs/2023-03-26-12-01-38/',
                          filename='MNIST_badnets_10_5_ScaleTrue4.csv',
                          selected_rows=[1])
                 ),
         csv_title='',
         legends=[
             '攻击者比例$\mathrm{ - 0\% }$',
             '攻击者比例$\mathrm{ - 20\% }$',
             '攻击者比例$\mathrm{ - 40\% }$',
         ],
         colors=['b', 'r', 'r'],
         linestyles=['-', '-', '--'],
         xlabel='轮次',
         ylabel='准确率')

    plot(y_data=(get_data(work_dir='poison/logs/2023-03-26-11-18-23/',
                          filename='MNIST_badnets_10_5_ScaleTrue4.csv',
                          selected_rows=[2]) +
                 get_data(work_dir='poison/logs/2023-03-26-11-31-25/',
                          filename='MNIST_badnets_10_5_ScaleTrue4.csv',
                          selected_rows=[2]) +
                 get_data(work_dir='poison/logs/2023-03-26-12-01-38/',
                          filename='MNIST_badnets_10_5_ScaleTrue4.csv',
                          selected_rows=[2])
                 ),
         csv_title='',
         legends=[
             '攻击者比例$\mathrm{ - 0\% }$',
             '攻击者比例$\mathrm{ - 20\% }$',
             '攻击者比例$\mathrm{ - 40\% }$',
         ],
         colors=['b', 'r', 'r'],
         linestyles=['-', '-', '--'],
         xlabel='轮次',
         ylabel='损失值')


def poison_defense():
    """
    投毒恶意检测性能
    """
    y_labels = ['准确率', '精确率', '召回率', '$\mathrm{ F_1 }$分数']
    y_labels = ['准确率', ]
    for i in range(1):
        plot(
            y_data=(
                    get_data(work_dir='poison/logs/2023-03-26-16-25-34/',
                             filename='MNIST_badnets_24_18_ScaleTrue4.csv',
                             selected_rows=[i + 1]) +
                    get_data(work_dir='poison/logs/2023-03-27-10-22-54/',
                             filename='MNIST_badnets_24_18_ScaleTrue4.csv',
                             selected_rows=[i + 1]) +
                    get_data(work_dir='poison/logs/2023-03-27-15-01-01/',
                             filename='MNIST_badnets_24_18_ScaleTrue4.csv',
                             selected_rows=[i + 1])
            ),
            csv_title='',
            legends=[
                '无攻击',
                '攻击+无防护措施',
                '攻击+恶意检测方案',
            ],
            colors=['b', 'r', 'r'],
            linestyles=['-', '--', '-'],
            xlabel='轮次',
            ylabel=y_labels[i]
        )


def backdoor_baseline():
    """
    后门攻击基准实验
    """
    plot(y_data=(get_data(work_dir='backdoor/logs/',
                          filename='CIFAR10_resnet18_4_3_ScaleTrue4_2023-03-10-14-57-58_trigger1.csv',
                          selected_rows=[1, 3]) +
                 get_data(work_dir='backdoor/logs/',
                          filename='CIFAR10_resnet18_4_3_ScaleTrue4_2023-03-10-15-40-40_trigger1.csv',
                          selected_rows=[1, 3])),
         legends=['$\mathrm{MTA}$ 不攻击',
                  '$\mathrm{BTA}$ 不攻击',
                  '$\mathrm{MTA}$ 连续攻击',
                  '$\mathrm{BTA}$ 连续攻击'],
         colors=['b', 'r', 'b', 'r'],
         linestyles=['--', '--', '-', '-'],
         xlabel='轮次',
         ylabel='准确率')

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


# backdoor_baseline()]

# backdoor_defense()

# backdoor_dba('backdoor/logs/2023-03-31-16-01-02/',
#              'CIFAR10_resnet18_16_12_ScaleTrue2_trigger1.csv')
#
# backdoor_dba('backdoor/logs/2023-04-03-10-48-37/',
#              'CIFAR10_resnet18_16_12_ScaleTrue3_trigger1.csv')

backdoor_dba('backdoor/logs/2023-04-03-14-56-45/',
             'CIFAR10_resnet18_16_12_ScaleTrue3_trigger1.csv')

# poison_baseline()

# poison_defense()
