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
    plot(y_data=(get_data(work_dir='../backdoor/logs/',
                          filename='CIFAR10_resnet18_4_3_ScaleTrue4_2023-03-10-14-57-58_trigger1.csv',
                          selected_rows=[1, 3]) +
                 get_data(work_dir='../backdoor/logs/',
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

    pass
