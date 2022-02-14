from itertools import permutations

print([" ".join(combo) for combo in permutations("Good morning Rosa!".split(), 3)])

s = "Find textbooks with titles containing 'NLP', or 'natural' and 'language', or 'computational' and 'linguistics'. "
print(len(set(s.split())))

import numpy as np

print(np.arange(1, 12 + 1).prod())  # 阶乘
