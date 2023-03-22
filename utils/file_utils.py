import json
import os
import time

TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'


def prepare_operation(args, prefix):
    start_time = time.time()
    start_time_str = time.strftime(TIME_FORMAT, time.localtime())
    # 创建文件夹
    prefix = prefix + '/' + start_time_str
    os.makedirs(f'{prefix}')
    # 保存配置参数
    with open(f'{prefix}/params.json', 'wt') as f:
        json.dump(vars(args), f, indent=4)
    f.close()

    return start_time, start_time_str, prefix
