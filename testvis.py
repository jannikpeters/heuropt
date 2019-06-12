import matplotlib.pyplot as plt
import numpy as np
hypervol = []
with open('test_vis.f', 'r') as fp:
    for i in range(280):
        test = fp.readline()
        len, val = test.split(' ')
        hypervol.append((float(len), -float(val)))
nd = []

for kp_val, rent in hypervol:
    dominated = False
    for kp, re in hypervol:
        if kp > kp_val and re > rent:
            dominated = True
    if not dominated:
        nd.append((kp_val, rent))

nd = np.random.choice(nd, 100, replace=False)
plt.scatter(*zip(*nd))
plt.show()
