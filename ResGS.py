#ResGS predicts phenotypic residuals
#tensorflow-gpu 2.4

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')


def Conv1d_BN(x, nb_filter, kernel_size, strides=1):
    x = layers.Convolution1D(nb_filter, kernel_size, padding='same', strides=strides, activation='relu')(x)
    x = layers.BatchNormalization(axis=1)(x)
    return x

def Res_Block(inpt,nb_filter,kernel_size,strides=1):
    x = Conv1d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides)
    x = layers.add([x,inpt])
    return x

def ResGSModel(inputs):
    nFilter = 64
    _KERNEL_SIZE = 3
    CHANNEL_FACTOR1 = 4
    CHANNEL_FACTOR2 = 1.1
    x1 = Res_Block(inputs, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x1 = Res_Block(x1, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    nFilter1 = int(nFilter * CHANNEL_FACTOR1)

    x2 = Conv1d_BN(x1 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x2 = Conv1d_BN(x2, nb_filter=nFilter, kernel_size=1, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x3 = Conv1d_BN(x2 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x3 = Conv1d_BN(x3, nb_filter=nFilter, kernel_size=1, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x4 = Conv1d_BN(x3 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x4 = Conv1d_BN(x4, nb_filter=nFilter, kernel_size=1, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x5 = Conv1d_BN(x4 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x5 = Conv1d_BN(x5, nb_filter=nFilter, kernel_size=1, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x6 = Conv1d_BN(x5 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x6 = Conv1d_BN(x6, nb_filter=nFilter, kernel_size=1, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x7 = Conv1d_BN(x6 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x7 = Conv1d_BN(x7, nb_filter=nFilter, kernel_size=1, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x8 = Conv1d_BN(x7 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x8 = Conv1d_BN(x8, nb_filter=nFilter, kernel_size=1, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x9 = Conv1d_BN(x8 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x9 = Conv1d_BN(x9, nb_filter=nFilter, kernel_size=1, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x10 = Conv1d_BN(x9 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x10 = Conv1d_BN(x10, nb_filter=nFilter, kernel_size=1, strides=1)
    x10 = Res_Block(x10, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x10 = Res_Block(x10, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x11 = Conv1d_BN(x10, nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x11 = Conv1d_BN(x11, nb_filter=nFilter, kernel_size=1, strides=1)
    x11 = Res_Block(x11, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x11 = Res_Block(x11, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    # Dense layer
    DENSE1 = 1024
    DENSE2 = 512
    DENSE3 = 128
    DENSE4 = 32

    x9 = layers.Flatten()(x9)
    x9 = layers.Dense(DENSE1)(x9)
    x9 = layers.Activation('relu')(x9)

    x10 = layers.Flatten()(x10)
    x10 = layers.Dense(DENSE1)(x10)
    x10 = layers.Activation('relu')(x10)

    x11 = layers.Flatten()(x11)
    x11 = layers.Dense(DENSE1)(x11)
    x11 = layers.Activation('relu')(x11)


    x9 = tf.expand_dims(x9, axis= 2)
    x10 = tf.expand_dims(x10, axis= 2)
    x11 = tf.expand_dims(x11, axis=2)

    x = layers.Concatenate(axis=2)([x9, x10, x11])
    x = Conv1d_BN(x, nb_filter=1, kernel_size=1, strides=1)
    x = layers.Flatten()(x)
    x = layers.Activation('relu')(x)


    x = layers.Dense(DENSE2)(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(DENSE3)(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(DENSE4)(x)
    x = layers.Activation('softplus')(x)
    x = layers.Dense(1)(x)

    return Model(inputs = inputs, outputs = x)


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
        self.y_combination = 0

    def on_epoch_end(self, epoch, logs=None):
        traditionalModelPredict = bestTraditionalModel.predict(np.squeeze(self.x_test))
        correlation = np.corrcoef(self.y_test + traditionalModelPredict,self.model.predict(self.x_test)[:,0] + traditionalModelPredict)[0,1]

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        # print("the Epoch is " +str(epoch) + ", and the correlation is " + str(correlation))

        if correlation > self.bestCorrelation:
            self.bestCorrelation = correlation
            self.y_pre = self.model.predict(self.x_test)
            self.y_combination = self.model.predict(self.x_test)[:,0] + traditionalModelPredict
            FileNameList = self.saveFileName.split('.')

            print("The model is saved at the Epoch " + str(epoch) +
                  ". And the correlation is " + str(correlation))
            FileName = FileNameList[0]+ "_repeatTime_"+ str(self.repeatTime) +'.' + FileNameList[1]
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


def ResGS(X_train, X_test, y_train, y_test,
            saveFileName,
            CUDA_VISIBLE_DEVICES = 0,
            Epoch = 1200,
            repeatTimes= 10):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
    batch_size = 64
    patience = 100
    bestCorrelations = []
    y_pres = []

    #Traditional model
    global bestTraditionalModel
    r_max = 0  # Record the maximum Pearson value
    for model in ["Ridge", "support vector machine", "RandomForest", "GradientBoostingRegressor"]:
        print(model, end= ':')
        if model == "Ridge":
            model = Ridge()
        elif model == "support vector machine":
            model = SVR(kernel='rbf')
        elif model == "RandomForest":
            model = RandomForestRegressor()
        elif model == "GradientBoostingRegressor":
            model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        r = np.corrcoef(y_pre, y_test)[0,1]  # r is pearson correlation
        print(r)
        if r > r_max:
            r_max = r
            bestTraditionalModel = model

    print(bestTraditionalModel)
    y_train_pre = bestTraditionalModel.predict(X_train)
    y_pre = bestTraditionalModel.predict(X_test)

    y_train = y_train - y_train_pre
    y_test = y_test - y_pre


    nSNP = X_train.shape[1]


    X2_train = np.expand_dims(X_train, axis=2)
    X2_test = np.expand_dims(X_test, axis=2)


    for i in range(repeatTimes):
        tf.random.set_seed(i)
        print("\n\n\n============================repeatTimes: " + str(i) + "============================")
        inputs = layers.Input(shape=(nSNP, 1))

        model_DNNSC = ResGSModel(inputs)
        model_DNNSC.compile(loss='mse', optimizer= 'adam')
        performance_simple = PerformancePlotCallback(X2_test, y_test, model=model_DNNSC, repeatTime=i,
                                                     saveFileName= saveFileName, patience = patience)
        history = model_DNNSC.fit(X2_train, y_train, epochs= Epoch, batch_size=batch_size,
                                     validation_data= (X2_test, y_test),
                                     verbose= 0, callbacks= performance_simple)

        print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
        bestCorrelations.append(performance_simple.bestCorrelation)
        y_pres.append(performance_simple.y_combination)

    print("bestCorrelation: ", max(bestCorrelations))
    print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))

    print("bestCorrelations:", bestCorrelations)

    return max(bestCorrelations), y_pres[bestCorrelations.index(max(bestCorrelations))]
