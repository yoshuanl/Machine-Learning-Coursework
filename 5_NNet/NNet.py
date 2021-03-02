#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""
import os
import sys
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

folder = sys.path[0] + "/"
os.chdir(folder)


""" Return data with bias term (1) and also balance data if needed """
def readAllPgm(file_catalog, byteorder = '>', balance = False):
    X = np.array([]).reshape(0, 960)
    y = np.array([], dtype = int).reshape(0, 1)
    filename = file_catalog.readline()[:-1]
    while filename:
        with open(filename, 'rb') as f:
            buffer = f.read()
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        new_pgm = np.frombuffer(buffer, dtype='u1' if int(maxval) < 256 else byteorder+'u2', 
                                count = int(width) * int(height), offset = len(header)).reshape(1, 960)
        
        X = np.r_[X, new_pgm]
        
        # y is 1 if the word "down" is in its file name
        is_down = (re.match(".*down.*", filename) != None)
        label = np.array([int(is_down)]).reshape(1,1)
        y = np.r_[y, label.T]
        
        if balance and is_down:
            # triple the amount of training data with label = 1
            y = np.r_[y, label.T, label.T]
            X = np.r_[X, new_pgm, new_pgm]
            
        filename = file_catalog.readline()[:-1]
        
    X = np.c_[np.ones(np.shape(X)[0]), X] # add bias term
    return X, y


    
class NeuralNetwork():
    def __init__(self):
        self.total_grids = 960
        self.hidden_perceptrons = 99
        self.epochs = 1000
        self.learning_rate = 0.1
    

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    
    def sigmoidDerivative(self, sig):
        return sig * (1 - sig)
    
    
    """ Scale randomly initialized weight into (-0.01, 0.01)"""
    def randomRange(self, org_rand):
        return ((org_rand * 2) - 1) * 0.01
    
    
    """ A data iterator picking sample without replacement """
    def batchWithoutReplacement(self, batch_size, train_X, train_y):
        N = np.shape(train_X)[0]
        idx = list(range(N))
        np.random.seed(7)
        random.shuffle(idx)
        for batch_i, i in enumerate(range(0, N, batch_size)):
            j = idx[i: min(i + batch_size, N)]
            yield batch_i, train_X.take(j, axis = 0), train_y.take(j, axis = 0)
    
    
    """ A data iterator picking sample with replacement """
    def batchWithReplacement(self, batch_size, train_X, train_y):
        N = np.shape(train_X)[0]
        for batch_i, i in enumerate(range(0, N, batch_size)):
            #j = nd.array(idx[i: min(i + batch_size, num_examples)])
            j = random.sample(range(N), batch_size)
            yield batch_i, train_X.take(j, axis = 0), train_y.take(j, axis = 0)
            
    
    def vanillaGD(self, train_X, train_y, test_X, test_y):
        np.random.seed(22)
        
        hidden_weights = self.randomRange(np.random.rand(np.shape(train_X)[1], self.hidden_perceptrons)) # include weight of bias term
        output_weights = self.randomRange(np.random.rand(self.hidden_perceptrons + 1, 1))
        input_data, labels = train_X, train_y
        N = np.shape(input_data)[0]
        
        loss_record = list()
        for epoch in range(self.epochs):
            # S stands for score
            hidden_S = np.dot(input_data, hidden_weights)
            hidden_X = self.sigmoid(hidden_S)
            
            # takes previous layer's output as input
            hidden_X = np.c_[np.ones(N), hidden_X]
            output_S = np.dot(hidden_X, output_weights)
            output_X = self.sigmoid(output_S)
            
            err = labels - output_X
            loss = sum(err ** 2) / N
            loss_record.append(loss)
            
            # backpropogation
            output_delta = (-1) * err / N * self.sigmoidDerivative(output_X)
            output_w_gradient = np.dot(hidden_X.T, output_delta)
            
            hidden_delta = self.sigmoidDerivative(hidden_X[:, 1:]) * np.dot(output_delta, output_weights.T[:, 1:])
            hidden_w_gradient = np.dot(input_data.T, hidden_delta)
        
            output_weights -= self.learning_rate * output_w_gradient
            hidden_weights -= self.learning_rate * hidden_w_gradient
        
        hidden_pred = self.sigmoid(np.dot(test_X, hidden_weights))
        hidden_pred = np.c_[np.ones(np.shape(hidden_pred)[0]), hidden_pred]
        prediction = self.sigmoid(np.dot(hidden_pred, output_weights))
        
        test_N = np.shape(test_y)[0]
        prediction_label = list(map(lambda x: int(x > 0.5), prediction))
        pred_err = (np.array(prediction_label).reshape(83, 1) - test_y) ** 2
        accuracy = (test_N - int(sum(pred_err)))/ test_N
        
        return accuracy, loss_record, pred_err
                
                
    def miniBatchGD(self, train_X, train_y, test_X, test_y, batch_size):
        np.random.seed(22)
        
        hidden_weights = self.randomRange(np.random.rand(np.shape(train_X)[1], self.hidden_perceptrons)) # include weight of bias term
        output_weights = self.randomRange(np.random.rand(self.hidden_perceptrons + 1, 1))
        
        loss_record = list()
        
        for epoch in range(self.epochs):
            loss_batch = list()
            for batch_i, input_data, labels in self.batchWithReplacement(batch_size, train_X, train_y):
                this_batch_size = np.shape(input_data)[0]
                # S stands for score
                hidden_S = np.dot(input_data, hidden_weights)
                hidden_X = self.sigmoid(hidden_S)
                
                # takes previous layer's output as input
                hidden_X = np.c_[np.ones(np.shape(hidden_X)[0]), hidden_X]
                output_S = np.dot(hidden_X, output_weights)
                output_X = self.sigmoid(output_S)
                
                err = labels - output_X
                loss = sum(err ** 2) / this_batch_size
                loss_batch.append(loss)
                
                # backpropogation
                output_delta = (-1) * err / this_batch_size * self.sigmoidDerivative(output_X)
                output_w_gradient = np.dot(hidden_X.T, output_delta)
                
                hidden_delta = self.sigmoidDerivative(hidden_X[:, 1:]) * np.dot(output_delta, output_weights.T[:, 1:])
                hidden_w_gradient = np.dot(input_data.T, hidden_delta)
                
                output_weights -= self.learning_rate * output_w_gradient
                hidden_weights -= self.learning_rate * hidden_w_gradient
                
            loss_record.append(np.average(loss_batch))
        
        hidden_pred = self.sigmoid(np.dot(test_X, hidden_weights))
        hidden_pred = np.c_[np.ones(np.shape(hidden_pred)[0]), hidden_pred]
        prediction = self.sigmoid(np.dot(hidden_pred, output_weights))
        
        test_N = np.shape(test_y)[0]
        prediction_label = list(map(lambda x: int(x > 0.5), prediction))
        pred_err = (np.array(prediction_label).reshape(83, 1) - test_y) ** 2
        accuracy = (test_N - int(sum(pred_err)))/ test_N
        
        return accuracy, loss_record, pred_err
    
    
    """ Plot losses over epochs """
    def plotLoss(self, loss_record, title, label):
        plt.ion()
        x = range(len(loss_record))
        y = loss_record
        plt.plot(x, y, label = label)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


    """ Print out Wrong Classification on Test Data """
    def printWrongCases(self, pred_err, test_X):
        N = np.shape(test_X)[0]
        for i in range(N):
            if pred_err[i]:
                plt.imshow(test_X[i, 1:].reshape(30, 32), cmap = 'gray', vmin = 0, vmax = 255)
                plt.show()
                print("Wrong Classification For Test Data idx.{}".format(i))
    
    


if __name__ == "__main__":
    print("training neural network...... ")
    # read test data
    testingfile_catalog = open("downgesture_test.list.txt")
    test_X, test_y = readAllPgm(testingfile_catalog)
    
    
    """ Vanilla Gradient Descent """
    trainingfile_catalog = open("downgesture_train.list.txt")
    train_X, train_y = readAllPgm(trainingfile_catalog, balance = True)

    NN = NeuralNetwork()
    accuracy, losses, prediction_error = NN.vanillaGD(train_X, train_y, test_X, test_y)
    print("\n ----------Vanilla Gradient Descent Result----------")
    print("Accuracy on Test Data: {:.2%}".format(accuracy))
    NN.plotLoss(losses, "Neural Network Loss Over Epochs", "Vanilla Gradient Descent")
    #NN.printWrongCases(prediction_error, test_X)
    
    
    """ Mini-batch Gradient Descent """
    trainingfile_catalog = open("downgesture_train.list.txt")
    train_X, train_y = readAllPgm(trainingfile_catalog, balance = False)

    NN = NeuralNetwork()
    batch_size = 128
    accuracy, losses, prediction_error = NN.miniBatchGD(train_X, train_y, test_X, test_y, batch_size)
    print("\n ----------Mini-batch Gradient Descent Result----------")
    print("Accuracy on Test Data: {:.2%}".format(accuracy))
    NN.plotLoss(losses, "Neural Network Loss Over Epochs", "Mini-batch Gradient Descent")
    #NN.printWrongCases(prediction_error, test_X)
    

    """ sklearn package 
    clf = MLPClassifier(solver = 'sgd', activation = 'logistic', batch_size = 128, max_iter = 1000, learning_rate_init = 0.1, learning_rate = 'adaptive', random_state = 22)
    clf.fit(train_X, train_y)
    clf.predict(test_X)"""


    # for avoiding the output graph disapear immediately after finishing execution
    print("\n<--------------------Execution End-------------------->")
    input("Press [enter] to close all graph windows.\n")