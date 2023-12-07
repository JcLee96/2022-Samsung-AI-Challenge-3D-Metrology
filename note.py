import math
import numpy as np
import matplotlib.pyplot as plt

import math
import numpy as np
import matplotlib.pyplot as plt


K = 6
value_range = 0.9

DDD = []
DDD_255 = []
for j in range(K+1):
    value = 1 - (np.exp(np.log(2) * (1 - j / K)) - 1)
    DDD.append(round(value, 3))
    value = math.ceil(value * 30)
    DDD_255.append(value)
print("Decreasing")
#DDD.reverse()
#DDD_255.reverse()
print(DDD_255)
DDD = []
DDD_255 = []
for j in range(K+1):
    value = 1 - (np.exp(np.log(2) * (1 - j / K)) - 1)
    DDD.append(round(value, 3))
    value = 30 - (math.ceil(value * 30))
    DDD_255.append(value)
print("Increasing")
#DDD.reverse()
DDD_255.reverse()
print(DDD_255)

DDD = []
DDD_255 = []
for j in range(K+1):
    value = (1 / K) * j
    value = int(round(value * 30, 0))
    DDD_255.append(value)
print("Uniform")
print(DDD_255)
#
# DDD = []
# DDD_255 = []
# K = 15
# value_range = 0.9
#
# for j in range(K+1):
#     value = np.exp(np.log(2) * (1 - j / K)) - 1
#     DDD.append(round(value, 3))
#     value = int(round(value * 255, 0))
#     DDD_255.append(value)
# print("DDD")
# print(DDD)
# print(DDD_255)
# alpha = -1
# beta = 1
# alpha_hat = alpha + abs(alpha)
# beta_hat = beta + abs(alpha)
#
# SDD = []
# SDD_255 = []
# for k in range(K+1):
#     value = ((math.sqrt(alpha_hat**2 + (beta_hat**2 - alpha_hat**2)*k/K) - abs(alpha)) * -1) * value_range
#     SDD.append(round(value, 3))
#     value = int(round(value * 141.667 + 127.5, 0))
#     SDD_255.append(value)
# print("SDD")
# print(SDD)
# print(SDD_255)
#
