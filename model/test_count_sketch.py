import numpy as np, random

"""
x = [batch_size, origin_dim]
A = [origin_dim, proj_dim]
y = [batch_size, proj_dim]
"""


def f():
    A = np.zeros([5,3])
    for i in range(5):
        A[i][random.randint(0, 2)] = 2 * random.randint(0, 1) - 1
    print A
    print ""

for i in range(5):
    f()
