import os
import pdb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 

from geo_tmain_webgnn import arg_parse, build_geowebgnn_model

class PlotMSECorr():
    def __init__(self):
        pass

    def rebuild_loss_pearson(self, path, epoch_num):
        test_epoch_loss_list = []
        train_epoch_loss_list = []
        test_epoch_pearson_list = []
        train_epoch_pearson_list = []
        min_test_loss = 100
        min_train_loss = 100
        min_test_id = 0
        for i in range(1, epoch_num + 1):
            # TEST LOSS
            test_df = pd.read_csv(path + '/TestPred' + str(i) + '.txt', delimiter=',')
            test_score_list = list(test_df['Score'])
            test_pred_list = list(test_df['Pred Score'])
            test_epoch_loss = mean_squared_error(test_score_list, test_pred_list)
            test_epoch_loss_list.append(test_epoch_loss)
            test_epoch_pearson = test_df.corr(method = 'pearson')
            test_epoch_pearson_list.append(test_epoch_pearson['Pred Score'][0])
            # TRAIN LOSS
            train_df = pd.read_csv(path + '/TrainingPred_' + str(i) + '.txt', delimiter=',')
            train_score_list = list(train_df['Score'])
            train_pred_list = list(train_df['Pred Score'])
            train_epoch_loss = mean_squared_error(train_score_list, train_pred_list)
            train_epoch_loss_list.append(train_epoch_loss)
            train_epoch_pearson = train_df.corr(method = 'pearson')
            train_epoch_pearson_list.append(train_epoch_pearson['Pred Score'][0])
            if test_epoch_loss < min_test_loss:
                min_test_loss = test_epoch_loss
                min_train_loss = train_epoch_loss
                min_test_id = i
        best_train_df = pd.read_csv(path + '/TrainingPred_' + str(min_test_id) + '.txt', delimiter=',')
        best_train_df.to_csv(path + '/BestTrainingPred.txt')
        best_test_df = pd.read_csv(path + '/TestPred' + str(min_test_id) + '.txt', delimiter=',')
        best_test_df.to_csv(path + '/BestTestPred.txt')
        # import pdb; pdb.set_trace()
        print('-------------BEST MODEL ID:' + str(min_test_id) + '-------------')
        print('BEST MODEL TRAIN LOSS: ', min_train_loss)
        print('BEST MODEL PEARSON CORR: ', train_epoch_pearson_list[min_test_id - 1])
        print('BEST MODEL TEST LOSS: ', min_test_loss)
        print('BEST MODEL PEARSON CORR: ', test_epoch_pearson_list[min_test_id - 1])
        #epoch_pearson_array_train = np.array(train_epoch_pearson_list)
        #epoch_loss_array_train = np.array(train_epoch_loss_list)
        epoch_pearson_array = np.array(test_epoch_pearson_list)
        epoch_loss_array = np.array(test_epoch_loss_list)
        #np.save(path + '/pearson.npy', epoch_pearson_array_train)
        #np.save(path + '/loss.npy', epoch_loss_array_train)
        np.save(path + '/test_pearson.npy', epoch_pearson_array)
        np.save(path + '/test_loss.npy', epoch_loss_array)
        return min_test_id

    def plot_loss_pearson(self, path, epoch_num):
        epoch_pearson_array = np.load(path + '/test_pearson.npy')
        epoch_loss_array = np.load(path + '/test_loss.npy')
        x = range(1, epoch_num + 1)
        plt.figure(1)
        plt.title('Training Loss and Pearson Correlation in ' + str(epoch_num) + ' Epochs') 
        plt.xlabel('Train Epochs') 
        plt.figure(1)
        plt.subplot(211)
        plt.plot(x, epoch_loss_array) 
        plt.subplot(212)
        plt.plot(x, epoch_pearson_array)
        plt.show()

    def plot_train_real_pred(self, path, best_model_num, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        pred_dl_input_df = pd.read_csv(path + '/TrainingPred_' + best_model_num + '.txt', delimiter = ',')
        pred_dl_input_df.to_csv(path + '/BestTrainingPred.txt', header=True, index=False)
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Training Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train'
        path = '../data-nci/plot/%s' % (file_name) + '.png'
        unit = 1
        if os.path.exists('./data/plot') == False:
            os.mkdir('../data-nci/plot')
        while os.path.exists(path):
            path = '../data-nci/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
        
    def plot_test_real_pred(self, path, best_model_num, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        pred_dl_input_df = pd.read_csv(path + '/TestPred' + best_model_num + '.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        pred_dl_input_df.to_csv(path + '/BestTestPred.txt', header=True, index=False)
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Test Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test'
        path = '../data-nci/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = '-./data-nci/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)

if __name__ == "__main__":
    ###########################################################################################
    ############### ANALYSE [MSE_LOSS/PEARSON CORRELATION] FROM RECORDED FILES ################
    ###########################################################################################
    path = '../data-nci/result/epoch_200'
    # path = './data/result/oneil_webgnn/epoch_500'
    epoch_num = 200
    min_test_id = PlotMSECorr().rebuild_loss_pearson(path, epoch_num)

    # BEST MODEL ID
    # fold 1: 184
    # fold 2: 331
    # fold 3: 61
    # fold 4: 467
    # fold 5: 136




    #PlotMSECorr().plot_loss_pearson(path, epoch_num)

    # # ANALYSE DRUG EFFECT
    # print('ANALYSING DRUG EFFECT...')
    epoch_time = '200'
    best_model_num = str(min_test_id)
    PlotMSECorr().plot_train_real_pred(path, best_model_num, epoch_time)
    PlotMSECorr().plot_test_real_pred(path, best_model_num, epoch_time)