import tensorflow as tf
import numpy as np
import scipy.io as sio

from tensorflow.keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def MSC_1DCNN(datashape=(8192,1) , class_number = 10):
    # 一维卷积神经网络 (CNN) 模型
    input_signal = Input(shape=(8192, 1))
    # 它采用形状为(8192, 1) 的输入信号，并应用具有不同内核大小和步幅的多个并行一维卷积层
    # 多个并行一维卷积层
    x1 = Conv1D(filters=1, kernel_size=2, padding='valid', strides=2)(input_signal)
    x1 = MaxPooling1D(pool_size=256)(x1)
    x2 = Conv1D(filters=1, kernel_size=4, padding='valid', strides=4)(input_signal)
    x2 = MaxPooling1D(pool_size=128)(x2)
    x3 = Conv1D(filters=1, kernel_size=8, padding='valid', strides=8)(input_signal)
    x3 = MaxPooling1D(pool_size=64)(x3)
    x4 = Conv1D(filters=1, kernel_size=16, padding='valid', strides=16)(input_signal)
    x4 = MaxPooling1D(pool_size=32)(x4)
    x5 = Conv1D(filters=1, kernel_size=32, padding='valid', strides=32)(input_signal)
    x5 = MaxPooling1D(pool_size=16)(x5)
    x6 = Conv1D(filters=1, kernel_size=64, padding='valid', strides=64)(input_signal)
    x6 = MaxPooling1D(pool_size=8)(x6)
    x7 = Conv1D(filters=1, kernel_size=128, padding='valid', strides=128)(input_signal)
    x7 = MaxPooling1D(pool_size=4)(x7)
    x8 = Conv1D(filters=1, kernel_size=256, padding='valid', strides=256)(input_signal)
    x8 = MaxPooling1D(pool_size=2)(x8)

    xx = concatenate([x1, x2, x3, x4, x5, x6, x7, x8], axis=-2)
    # 这些层的输出沿着最后一个轴连接起来。
    xx = Flatten()(xx)
    xx = Dense(128, activation='relu')(xx)
    # 连接的输出被展平并通过具有 128个单元和ReLU激活的密集层
    output = Dense(10, activation='softmax')(xx)
    # 包含10个单元和softmax激活的密集层，它表示每个类别的输出概率
    model = Model(inputs=input_signal, outputs=output)

    return model


def normalization_processing(data):
    data_mean = data.mean()
    data_var = data.var()

    data = data - data_mean
    data = data / data_var

    return data


my_model = MSC_1DCNN(datashape=(8192, 1), class_number = 10)

datafile_path = '/content/drive/MyDrive/Colab Notebooks/all_data_DriveEnd.mat'
data = sio.loadmat(datafile_path)
category_list = ['NM','BA_007','BA_014','BA_021','OR_007','OR_014','OR_021','IR_007','IR_014','IR_021']
# category_list = ['NM','BA_014','OR_014','IR_014']

X_train = []
y_train = []
X_test1  = []
y_test1  = []
sample_length = 8192

sample_num_each_signal = 100 # in training data or testing data, each full signal actually generates 200 samples.
start_idx = np.int64(np.round(np.linspace(0,60000-sample_length,sample_num_each_signal)))
# start_idx = list(range(0,60000-sample_length+1,round(60000/sample_num_each_signal)))
for i in range(10):
    this_ctgr = category_list[i]

    for j in range(4):# dongli
        key_name = this_ctgr + '_' + str(j)
        this_ctgr_data_j = data[key_name]

        [X_train.append(this_ctgr_data_j[k:k+sample_length]) for k in start_idx]
        [y_train.append(i) for k in start_idx]
        # y_train.append(i)
        [X_test1.append(this_ctgr_data_j[k+60001:k+60001+sample_length]) for k in start_idx]
        [y_test1.append(i) for k in start_idx]
        # y_test.append(i)


print(len (X_train), len(X_test1))
X_train = np.squeeze(np.array(X_train))
X_test1 = np.squeeze(np.array(X_test1))


X_valid, X_test, y_valid, y_test = train_test_split(X_test1, y_test1, test_size=0.75, random_state=42)


X_train = np.array([normalization_processing(data) for data in X_train])
X_train = np.expand_dims(X_train,axis=2)
y_train_one_hot = to_categorical(np.array(y_train), len(category_list))


X_test = np.array([normalization_processing(data) for data in X_test])
X_test = np.expand_dims(X_test,axis=2)
y_test_one_hot = to_categorical(np.array(y_test), len(category_list))


X_valid = np.array([normalization_processing(data) for data in X_valid])
X_valid = np.expand_dims(X_valid,axis=2)
y_valid_one_hot = to_categorical(np.array(y_valid), len(category_list))


# configure training
opt = tf.keras.optimizers.Adam(lr = 0.01)
my_model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.0000001)
history_callback = History()
history = my_model.fit(
    x = X_train,
    y = y_train_one_hot ,
    batch_size = 32,
    epochs = 300,# times
    validation_data = (X_valid, y_valid_one_hot),
    shuffle=True,
    verbose = 1,
    # verbose = 0
    callbacks=[history_callback]

)

y_pred = my_model.predict(X_test) #shape: 22313*7 #y_test:22313*1
y_pred = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_test, y_pred)

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']



