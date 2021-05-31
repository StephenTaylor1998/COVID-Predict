# %%

import pandas as pd

data = pd.read_csv("datasets/china.csv")

# %%

data.pop("统计日期")
data = data.values

# %%

import numpy as np

window = 30  # 窗口长度
max_ = np.max(data, axis=0, keepdims=True)
min_ = np.min(data, axis=0, keepdims=True)
data = (data - min_) / (max_ - min_)  # 归一化
# data_process=data_process*2-1
seq = []
target = []
for i in range(window + 1, len(data)):  # 构建数据集
    seq.append(data[i - window:i, :][np.newaxis, :, :])
    target.append(data[i:i + 1, :])
seq = np.concatenate(seq, axis=0)[:, :, :, np.newaxis]
target = np.concatenate(target, axis=0)
print(seq.shape, target.shape)

# %%

# #划分训练集和测试集
# test_size=0.2#测试集占2%
# X_train,X_test=seq[int(len(seq)*0.2):,:,:],seq[:int(len(seq)*0.2),:,:]
# y_train,y_test=target[int(len(seq)*0.2):],target[:int(len(seq)*0.2)]
from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test=train_test_split(seq,target,test_size=0.25,random_state=2021,shuffle=True)
test_size = 0.2
X_train, X_test = seq[int(len(seq) * 0.2):, :, :], seq[:int(len(seq) * 0.2), :, :]
y_train, y_test = target[int(len(seq) * 0.2):, :], target[:int(len(seq) * 0.2), :]
# print(type(X_train),type(y_train))
print(X_test)

# %%

from tensorflow.keras import layers, Sequential
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
log_dir = "log/CNN"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

tf.random.set_seed(2021)
db_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(10)
db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(10)

model = Sequential()
model.add(layers.Input(shape=(30, 3, 1), dtype=tf.float32))
model.add(layers.Conv2D(filters=20, kernel_size=(20, 2), strides=(1, 1), padding="same",
                        activation=tf.nn.leaky_relu))  # CNN，filters卷积核数量，kernel_size卷积核大小，strides步长，padding填充
model.add(layers.Conv2D(filters=10, kernel_size=(10, 2), strides=(1, 1), padding="same",
                        activation=tf.nn.leaky_relu))  # CNN，filters卷积核数量，kernel_size卷积核大小，strides步长，padding填充
model.add(
    layers.Conv2D(filters=20, kernel_size=(2, 2), strides=(1, 1), padding="same", activation=tf.nn.leaky_relu))  # CNN
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))  # 最大池化，pool_size池化核大小，stride步长，padding填充
model.add(layers.Flatten())  # 压平
model.add(layers.Dense(500, activation=tf.nn.leaky_relu))  # 全连接层，激活函数为leaky relu，输出大小为20
model.add(layers.Dense(100, activation=tf.nn.leaky_relu))  # 全连接层，激活函数为leaky relu，输出大小为20
model.add(layers.Dense(50, activation=tf.nn.leaky_relu))  # 全连接层，激活函数为leaky relu，输出大小为20
# model.add(layers.Dense(20,activation=tf.nn.leaky_relu))#全连接层，激活函数为leaky relu，输出大小为20
model.add(layers.Dense(3))  # 全连接层，输出大小为3
model.summary()  # 总打印模型信息
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.MSE,
              metrics=tf.keras.metrics.MeanAbsoluteError())  # 编译，Adam优化器，MSE损失
his = model.fit(db_train, validation_data=db_test, validation_freq=1, epochs=400, callbacks=[tensorboard])  # 训练

#%%

import  matplotlib.pyplot as plt
# his = model.fit(X_train, epochs=epochs, validation_data=val_ds)
# his =model.fit(db_train,validation_data=db_test,validation_freq=1,epochs=60,callbacks=[tensorboard])
# model.save_weights('./20epoch_(90%-2).h5', overwrite=True)

# 损失
fig, ax = plt.subplots(figsize=[5,7])
ax.plot(his.history['loss'], label='train_loss')
ax.plot(his.history['val_loss'], label='val_loss')
ax.legend()
ax.set_xlabel('epochs')
ax.set_ylabel('losses')
ax.set_title('CNN')
plt.show()

# %%

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# predict=model.predict(X_test)#预测
# predict=predict*(max_-min_)+min_#反归一化
# y=y_test*(max_-min_)+min_#反归一化
predict = model.predict(X_test)
# predict=predict*(max_-min_)+min_
predict[predict < 0] = 0
y = y_test
# y=y_test*(max_-min_)+min_
# "新增确诊	"
print(r2_score(y[:, 0], predict[:, 0]))  # R^2
print(mean_absolute_error(y[:, 0], predict[:, 0]))  # MAE
print(mean_absolute_percentage_error(y[:, 0], predict[:, 0]))  # MAPE
print(mean_squared_error(y[:, 0], predict[:, 0]))  # MSE
print()
# "新增治愈"
print(r2_score(y[:, 1], predict[:, 1]))
print(mean_absolute_error(y[:, 1], predict[:, 1]))
print(mean_absolute_percentage_error(y[:, 1], predict[:, 1]))
print(mean_squared_error(y[:, 1], predict[:, 1]))
print()
# "新增死亡"
print(r2_score(y[:, 2], predict[:, 2]))
print(mean_absolute_error(y[:, 1], predict[:, 1]))
print(mean_absolute_percentage_error(y[:, 1], predict[:, 1]))
print(mean_squared_error(y[:, 1], predict[:, 1]))
print()

# %%

predict = predict * (max_ - min_) + min_
y = y_test * (max_ - min_) + min_

# %%

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(predict[:, 0], label="predict")
plt.plot(y[:, 0], label="real")
plt.legend(loc="best")
plt.title("新增确诊")
plt.show()

# %%


plt.plot(predict[:, 1], label="predict")
plt.plot(y[:, 1], label="real")
plt.legend(loc="best")
plt.title("新增治愈")
plt.show()

# %%


plt.plot(predict[:, 2], label="predict")
plt.plot(y[:, 2], label="real")
plt.legend(loc="best")
plt.title("新增死亡")
plt.show()
# %%
