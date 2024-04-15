import numpy as np
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

global save_model

def DNNGPModel(inputs, nFilter, kernel_size = 3, strides = 1):
    x = layers.Convolution1D(filters = nFilter, kernel_size = kernel_size, padding= "same", strides= strides,
                             activation= "relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Convolution1D(filters = nFilter, kernel_size = kernel_size, padding= "same", strides= strides,
                             activation= "relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Convolution1D(filters= 1, kernel_size=kernel_size, padding="same", strides=strides,
                             activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return Model(inputs = inputs, outputs = x)


# def DNNGPModel(inputs, nFilter, kernel_size = 3, strides = 1):
#     x = layers.Convolution1D(filters = nFilter, kernel_size = kernel_size, padding= "same", strides= strides,
#                              activation= "relu")(inputs)
#     x = layers.Dropout(0.3)(x)
#     x = layers.BatchNormalization(axis=1)(x)
#     x = layers.Convolution1D(filters = nFilter, kernel_size = kernel_size, padding= "same", strides= strides,
#                              activation= "relu")(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.BatchNormalization(axis=1)(x)
#     x = layers.Convolution1D(filters= 1, kernel_size=kernel_size, padding="same", strides=strides,
#                              activation="relu")(x)
#     x = layers.BatchNormalization(axis=1)(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(3)(x)
#     x = layers.Dropout(0.03)(x)
#     x = layers.Dense(1)(x)
#
#     return Model(inputs = inputs, outputs = x)


# def DNNGPModel(inputs, nFilter, kernel_size = 3, strides = 1):
#     x = layers.Convolution1D(filters = nFilter, kernel_size = kernel_size, padding= "same", strides= strides,
#                              activation= "relu")(inputs)
#     x = layers.Dropout(0.3)(x)
#     x = layers.BatchNormalization(axis=1)(x)
#     x = layers.Convolution1D(filters = nFilter, kernel_size = kernel_size, padding= "same", strides= strides,
#                              activation= "relu")(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.BatchNormalization(axis=1)(x)
#     x = layers.Convolution1D(filters= 1, kernel_size=kernel_size, padding="same", strides=strides,
#                              activation="relu")(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.BatchNormalization(axis=1)(x)
#
#     x = layers.Flatten()(x)
#     x = layers.Dense(1)(x)
#
#     return Model(inputs = inputs, outputs = x)


class PerformancePlotCallback(keras.callbacks.Callback):
    '''
    Record each epoch result
    '''
    def __init__(self, x_test, y_test, model, repeatTime, saveFileName, patience = 300):
        super(PerformancePlotCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.repeatTime = repeatTime
        self.bestCorrelation = 0
        self.saveFileName = saveFileName
        self.patience = patience
        self.y_pre = 0
        self.wait = 0
        self.stopped_epoch = 0


    def on_epoch_end(self, epoch, logs=None):
        correlation = np.corrcoef(self.y_test,self.model.predict(self.x_test)[:,0])[0,1]

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        # print("the Epoch is " +str(epoch) + ", and the correlation is " + str(correlation))

        if correlation > self.bestCorrelation:
            self.bestCorrelation = correlation
            self.y_pre = self.model.predict(self.x_test)
            FileNameList = self.saveFileName.split('.')

            print("The model is saved at the Epoch " + str(epoch) +
                  ". And the correlation is " + str(correlation))
            self.save_model = self.model
            # FileName = FileNameList[0]+ "_repeatTime_"+ str(self.repeatTime) +'.' + FileNameList[1]
            # self.model.save(FileName)
            # y_true = self.y_test + traditionalModelPredict
            # y_traditionalModel = traditionalModelPredict
            # y_residual = self.model.predict(self.x_test)[:,0]
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_true.txt", y_true)
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_traditionalModel.txt", y_traditionalModel)
            # np.savetxt(FileNameList[0]+ str(self.repeatTime) + "_y_residual.txt", y_residual)
            #new_model = keras.models.load_model('path_to_my_model.h5')
            self.wait = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def DNNGP(X_train, X_test, y_train, y_test,
            saveFileName,
            CUDA_VISIBLE_DEVICES = 0,
            Epoch = 3000,):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
    nFilter = 10
    batch_size = 64
    patience = 1000
    bestCorrelations = []
    y_pres = []


    nSNP = X_train.shape[1]

    X2_train = np.expand_dims(X_train, axis=2)
    X2_test = np.expand_dims(X_test, axis=2)


    inputs = layers.Input(shape=(nSNP, 1))

    model = DNNGPModel(inputs= inputs, nFilter= nFilter)
    # model.compile(loss='mse', optimizer='adam')
    model.compile(loss='mae', optimizer='adam')
    performance_simple = PerformancePlotCallback(X2_test, y_test, model= model, repeatTime=0,
                                                     saveFileName=saveFileName, patience=patience)
    history = model.fit(X2_train, y_train, epochs=Epoch, batch_size=batch_size,
                                  validation_data=(X2_test, y_test),
                                  verbose=0, callbacks=performance_simple)

    print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
    bestCorrelations.append(performance_simple.bestCorrelation)
    y_pres.append(performance_simple.y_pre)

    print("bestCorrelation: ", max(bestCorrelations))
    # print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))

    print("bestCorrelations:", bestCorrelations)


    # epochs = range(len(history.history['acc']))
    # plt.figure()
    # plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
    # plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # plt.savefig('/root/notebook/help/figure/model_V3.1_loss.jpg')


    # # 绘制训练 & 验证的损失值
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    print("min(history.history['val_loss']):", min(history.history['val_loss']))
    print("history.history['val_loss'].index(min(history.history['val_loss'])):", history.history['val_loss'].index(min(history.history['val_loss'])))

    return max(bestCorrelations), y_pres[bestCorrelations.index(max(bestCorrelations))], performance_simple.save_model


def DNNGP_training_set(X_train, y_train,
            saveFileName,
            CUDA_VISIBLE_DEVICES = 0,
            Epoch = 3000,):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
    nFilter = 10
    batch_size = 64
    patience = 1000
    bestCorrelations = []
    y_pres = []

    nSNP = X_train.shape[1]

    rows = X_train.shape[0]

    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
    X_test = X_train[int(rows*0.8):]
    y_test = y_train[int(rows*0.8):]
    X_train = X_train[:int(rows * 0.8)]
    y_train = y_train[:int(rows * 0.8)]

    X2_train = np.expand_dims(X_train, axis=2)
    X2_test = np.expand_dims(X_test, axis=2)


    inputs = layers.Input(shape=(nSNP, 1))

    model = DNNGPModel(inputs= inputs, nFilter= nFilter)
    # model.compile(loss='mse', optimizer='adam')
    model.compile(loss='mae', optimizer='adam')
    performance_simple = PerformancePlotCallback(X2_test, y_test, model= model, repeatTime=0,
                                                     saveFileName=saveFileName, patience=patience)
    history = model.fit(X2_train, y_train, epochs=Epoch, batch_size=batch_size,
                                  validation_data=(X2_test, y_test),
                                  verbose=0, callbacks=performance_simple)

    print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
    bestCorrelations.append(performance_simple.bestCorrelation)
    y_pres.append(performance_simple.y_pre)

    print("bestCorrelation: ", max(bestCorrelations))
    # print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))

    print("bestCorrelations:", bestCorrelations)


    # epochs = range(len(history.history['acc']))
    # plt.figure()
    # plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
    # plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # plt.savefig('/root/notebook/help/figure/model_V3.1_loss.jpg')


    # # 绘制训练 & 验证的损失值
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    print("min(history.history['val_loss']):", min(history.history['val_loss']))
    print("history.history['val_loss'].index(min(history.history['val_loss'])):", history.history['val_loss'].index(min(history.history['val_loss'])))

    return max(bestCorrelations), y_pres[bestCorrelations.index(max(bestCorrelations))], performance_simple.save_model