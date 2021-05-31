import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data = pd.read_csv("datasets/china.csv")

# data_process.pop("统计日期")
data = data.values[:, : 3]
# plt.plot(data_process[:, 0])
# # plt.show()
# plt.plot(data_process[:, 1])
# # plt.show()
# plt.plot(data_process[:, 2])
# plt.show()

window = 30  # 窗口长度

inputs = []
target = []
for i in range(window + 1, len(data)):  # 构建数据集
    inputs.append(data[i - window:i, :])
    target.append(data[i:i + 1, :])

inputs = numpy.array(inputs)
target = numpy.array(target)

# seq = np.concatenate(seq, axis=0)[:, :, :, np.newaxis]
# target = np.concatenate(target, axis=0)
# test_size = 0.2
x_train, x_test = inputs[int(inputs.shape[0] * 0.2):, :], inputs[:int(inputs.shape[0] * 0.2), :]
y_train, y_test = target[int(target.shape[0] * 0.2):, :], target[:int(target.shape[0] * 0.2), :]
#
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

