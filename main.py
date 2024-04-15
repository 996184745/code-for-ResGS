from __future__ import print_function
import argparse
import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingWarmRestarts
# from torch.optim import lr_scheduler
import tensorflow as tf

from dataloader import dataset_loader
# from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()


# from DNNSCWithTraditionalModelNoAnnotate import DNNSCWithTraditionalModel
from ResGS import ResGS, ResGSWithTraditionalModel, ResGS_pure
from DNNGP import DNNGP
import os
import errno
import shutil
import os.path as osp

import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    # Amylose content    Alkali spreading value     Protein content
    # traintest_F5_zhugao.csv
    # "44kgwas/PhenoGenotype_Protein content16000.csv"
    # data_file= "rice395/PhenoGenotype_HULLED.SEED.LENGTH.csv"
    # data_file = "rice413/PhenoGenotype_Seed number per panicleall.csv"
    # data_file = "wheat599/PhenoGenotype4OtherEnvironment.csv"
    "maize/PhenoGenotype_EarHT.csv"

    data_files = ["maize/PhenoGenotype_dpoll.csv",
                  "maize/PhenoGenotype_EarDia.csv",
                  "maize/PhenoGenotype_EarHT.csv",]


    # data_files = ["rice413/PhenoGenotype_Alkali spreading valueall.csv",
    # "rice413/PhenoGenotype_Amylose contentall.csv",
    # "rice413/PhenoGenotype_Panicle number per plantall.csv",
    # "rice413/PhenoGenotype_Protein contentall.csv",
    # "rice413/PhenoGenotype_Seed lengthall.csv",
    # "rice413/PhenoGenotype_Seed number per panicleall.csv",]


    # data_files = ["rice395/PhenoGenotype_Amylose.Content.csv",
    # "rice395/PhenoGenotype_SEED.LENGTH.csv"]

    # data_files = ["wheat599/PhenoGenotype1OtherEnvironment.csv",
    # "wheat599/PhenoGenotype2OtherEnvironment.csv",
    # "wheat599/PhenoGenotype3OtherEnvironment.csv",
    # "wheat599/PhenoGenotype4OtherEnvironment.csv",]

    # data_files = ["wheat599/PhenoGenotype1.csv",
    #               "wheat599/PhenoGenotype2.csv",
    #               "wheat599/PhenoGenotype3.csv",
    #               "wheat599/PhenoGenotype4.csv", ]

    model_select = ResGSWithTraditionalModel

    for data_file in data_files:
        GS_run(data_file, model_select)




def GS_run(data_file, model_select):
    dta_ds = dataset_loader(data_file=data_file)
    print("===============data_file: ", data_file)

    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    best_pearn_list = []
    best_mae_list = []

    for fold, (train_index, test_index) in enumerate(skfold.split(dta_ds, dta_ds.label)):
        print("\n\n\n============================fold: " + str(fold) + "============================")

        # ==============set train size======================
        # trainSize = 100
        # np.random.seed(0)
        # print("len(set(train_index)): ", len(set(train_index)))
        # print("len(train_index): ", len(train_index))
        # temp_index = np.random.choice(train_index, trainSize, replace = False) #不重复抽样
        # left_index = np.setdiff1d(train_index, temp_index)
        # train_index = temp_index
        # test_index = np.union1d(test_index, left_index)
        # ==============set train size======================

        X_train = dta_ds.ppg[[train_index]]
        y_train = dta_ds.hr[[train_index]]
        X_test = dta_ds.ppg[[test_index]]
        y_test = dta_ds.hr[[test_index]]

        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        # support vector machine
        # svr_rbf = SVR(kernel='rbf')
        # y_pre = svr_rbf.fit(X_train, y_train).predict(X_test)

        # RRBLUP
        # ridge = Ridge()
        # y_pre = ridge.fit(X_train, y_train).predict(X_test)

        # # RandomForest
        # RFR = RandomForestRegressor()
        # y_pre = RFR.fit(X_train, y_train).predict(X_test)

        # # GradientBoostingRegressor
        # GBR = GradientBoostingRegressor()
        # y_pre = GBR.fit(X_train, y_train).predict(X_test)

        # DNNSC
        # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        # print(gpus, cpus)
        # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        # DNNGP
        # r, y_pre = DNNGP(X_train, X_test, y_train, y_test, "largeModel",
        #                  CUDA_VISIBLE_DEVICES=0,
        #                  Epoch=3000,)

        # ResGS
        # r, y_pre = ResGS(X_train, X_test, y_train, y_test, "largeModel",
        #                  CUDA_VISIBLE_DEVICES=0,
        #                  Epoch=3000,
        #                  repeatTimes= 2)
        r, y_pre = model_select(X_train, X_test, y_train, y_test, "largeModel",
                                             CUDA_VISIBLE_DEVICES=0,
                                             Epoch=2000,
                                             repeatTimes=3)
        # r, y_pre, _ = ResGS_pure(X_train, X_test, y_train, y_test, "largeModel",
        #                  CUDA_VISIBLE_DEVICES=0,
        #                  Epoch=3000,)
        # print("first r:",r)
        # print("y_pre:", y_pre)
        # print("y_test:", y_test)

        # r, p = stats.pearsonr(y_pre, y_test)
        # print("second r:", r)

        MAE = mean_absolute_error(y_pre, y_test)

        best_pearn_list.append(r)
        best_mae_list.append(MAE)
        print("===============data_file: ", data_file)
    #     # if args.save_model:
    #     #     torch.save(model.state_dict(), "mnist_cnn.pt")
    #
    mean_best_pearn = np.mean(best_pearn_list)
    mean_best_mae = np.mean(best_mae_list)
    for i, value in enumerate(best_pearn_list):
        print("{} cross-validation best pearn:{}".format(i, value))
    for i, value in enumerate(best_mae_list):
        print("{} cross-validation best MAE:{}".format(i, value))
    print("Mean Best <Pearn>:{}".format(mean_best_pearn))
    print("Mean Best <MAE>:{}".format(mean_best_mae))

    print('finished')

    # sns.scatterplot(
    #     x=y_pre, y=y_test
    # )
    # plt.show()



def main2():
    # Amylose content    Alkali spreading value     Protein content
    # traintest_F5_zhugao.csv
    # "44kgwas/PhenoGenotype_Protein content16000.csv"
    # data_file= "rice395/PhenoGenotype_HULLED.SEED.LENGTH.csv"
    data_file = "wheat599/PhenoGenotypeInteraction.csv"
    env_file = "wheat599/env.csv"
    # data_file = "maize/PhenoGenotype_EarHT.csv"

    dta_ds = dataset_loader(data_file= data_file)
    env = pd.read_csv(env_file, header= None, index_col= None)
    env = np.array(env)
    print("===============data_file: ", data_file)


    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    best_pearn_list = []
    best_mae_list = []

    for fold, (train_index, test_index) in enumerate(skfold.split(dta_ds, dta_ds.label)):
        print("\n\n\n============================fold: " + str(fold) + "============================")
        X_train = dta_ds.ppg[[train_index]]
        y_train = dta_ds.hr[[train_index]]
        X_test = dta_ds.ppg[[test_index]]
        y_test = dta_ds.hr[[test_index]]


        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        env_train = np.squeeze(env[train_index])
        env_test = np.squeeze(env[test_index])

        y_train_env1 = []
        y_train_env2 = []
        y_train_env3 = []
        y_train_env4 = []

        for i in range(len(y_train)):
            if env_train[i] == "env1":
                y_train_env1.append(y_train[i])
            elif env_train[i] == "env2":
                y_train_env2.append(y_train[i])
            elif env_train[i] == "env3":
                y_train_env3.append(y_train[i])
            elif env_train[i] == "env4":
                y_train_env4.append(y_train[i])

        meanEnv1 = np.mean(y_train_env1)
        meanEnv2 = np.mean(y_train_env2)
        meanEnv3 = np.mean(y_train_env3)
        meanEnv4 = np.mean(y_train_env4)

        for i in range(len(y_train)):
            if env_train[i] == "env1":
                y_train[i] = y_train[i] - meanEnv1
            elif env_train[i] == "env2":
                y_train[i] = y_train[i] - meanEnv2
            elif env_train[i] == "env3":
                y_train[i] = y_train[i] - meanEnv3
            elif env_train[i] == "env4":
                y_train[i] = y_train[i] - meanEnv4

        for i in range(len(y_test)):
            if env_test[i] == "env1":
                y_test[i] = y_test[i] - meanEnv1
            elif env_test[i] == "env2":
                y_test[i] = y_test[i] - meanEnv2
            elif env_test[i] == "env3":
                y_test[i] = y_test[i] - meanEnv3
            elif env_test[i] == "env4":
                y_test[i] = y_test[i] - meanEnv4

        # for singleEnv in set(env):
        #     print(singleEnv)

        # support vector machine
        # svr_rbf = SVR(kernel='rbf')
        # y_pre = svr_rbf.fit(X_train, y_train).predict(X_test)

        # RRBLUP
        # ridge = Ridge()
        # y_pre = ridge.fit(X_train, y_train).predict(X_test)

        # # RandomForest
        # RFR = RandomForestRegressor()
        # y_pre = RFR.fit(X_train, y_train).predict(X_test)

        # # GradientBoostingRegressor
        # GBR = GradientBoostingRegressor()
        # y_pre = GBR.fit(X_train, y_train).predict(X_test)

        #DNNSC
        # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        # print(gpus, cpus)
        # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


        r, y_pre = DNNSCWithTraditionalModel(X_train, X_test, y_train, y_test, "largeModel",
                         saveFileName = 'output/checkpoint_'+str(fold)+'.h5',
                         CUDA_VISIBLE_DEVICES=4,
                         Epoch=1200,
                         repeatTimes= 10)




        r, p = stats.pearsonr(y_pre, y_test)
        MAE = mean_absolute_error(y_pre, y_test)

        best_pearn_list.append(r)
        best_mae_list.append(MAE)
        print("===============data_file: ", data_file)
    #     # if args.save_model:
    #     #     torch.save(model.state_dict(), "mnist_cnn.pt")
    #
    mean_best_pearn = np.mean(best_pearn_list)
    mean_best_mae = np.mean(best_mae_list)
    for i, value in enumerate(best_pearn_list):
        print("{} cross-validation best pearn:{}".format(i, value))
    for i, value in enumerate(best_mae_list):
        print("{} cross-validation best MAE:{}".format(i, value))
    print("Mean Best <Pearn>:{}".format(mean_best_pearn))
    print("Mean Best <MAE>:{}".format(mean_best_mae))

    print('finished')

if __name__ == '__main__':
    main()

