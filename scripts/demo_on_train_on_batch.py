import tensorflow as tf
import time
import sys
import os
import numpy as np
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# print(tf.__version__)
# print(sys.version_info)
# for module in mpl, np, pd, sklearn, tf, keras:
#     print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
# 归一化，x = (x - u) / std
scalar = StandardScaler()
# x_train: [None, 28, 28] -> [None, 784]
x_train_scaled = scalar.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scalar.fit_transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scalar.fit_transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)




class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Tensorboard, earlystopping, ModelCheckpoint 等回调函数
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    # 5 个epoch没有提升就停止训练； 提升小于1e-3视为没有提升
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]

# history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks)
# call back实验

for i, (x, y) in enumerate(zip(x_train_scaled, y_train)):
    x = tf.constant(x)
    print(x.shape)
    print(y)
    if i > 20:
        break
    log_train = model.train_step((x, y))
    print(log_train)