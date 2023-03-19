from itertools import combinations
import torch

nums = ['a', 'b', 'c', 'd']

for i, j in list(combinations(nums, 2)):
    print(f'{i}+{j}')

pass
