#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""

"""
Introduction:
    1. No absolute file path in this file. This script can be executed through terminal.
    
    2. I use the weight output by linear regression as initial weight 
    of my PLA and pocket algorithm to improve the performance.
    
    3. Performance comparison between using linear regression output weight
    and using zeros as initial weight is attached in my report.
"""

# If the script is run in python IDE,
# for enabling interactive 3D plot, please execute below line before plotting
# %matplotlib auto



import sys
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# read data
folder = sys.path[0] + "/"
#folder = "/Users/stanley/Documents/USC/Courses/Spring20_Courses/INF552/LinearClassification_Regression"


""" Evaluate the performance of weights"""
""" used by perceptron algorithm, pocket algorithm and logistic regression"""
class Evaluation():
    def __init__(self, points, answer):
        self.points = points
        self.answer = answer
        self.N = len(points)
        
    def evaluate(self, x, w):
        if x.dot(w) < 0:
            return -1
        else:
            return +1
        
    def countMisclassifiedPoints(self, weight):
        misclassification = 0
        for row, actual_classification in zip(self.points, self.answer):
            if self.evaluate(row, weight) != actual_classification:
                misclassification += 1
        return misclassification
      
    def calculateAccuracy(self, weight):
        error_count = self.countMisclassifiedPoints(weight)
        return (self.N - error_count)/ self.N
    
    

""" Include perceptron algorithm and pocket algorithm."""
class MyLinearClassification():
    def __init__(self, X, y, initial_weight, max_iteration = 7000):
        self.algorithm = ""
        x0 = np.ones(X.shape[0])
        self.X = np.c_[x0, X]
        self.y = y
        self.max_iteration = max_iteration
        #self.initial_weight = np.zeros(self.X.shape[1])
        self.initial_weight = initial_weight
        self.EV = Evaluation(self.X, self.y)
    
    
    def perceptron(self):
        if self.algorithm:
            print("Error Message: please reinitialize MyLinearClassification class")
            return
        self.algorithm = "Perceptron"
        
        # step 0: start from initial weight
        weight = self.initial_weight
        
        accuracy = 0
        rd = 0
        print("\n\n[Perceptron Algorithm Running...]")
        while accuracy != 1:
            rd += 1
            for row, actual_classification in zip(self.X, self.y):
                # step 1: find the next mistake of w
                if self.EV.evaluate(row, weight) != actual_classification:
                    # step 2: correct the mistake
                    weight = weight + actual_classification * row
            
            accuracy = self.EV.calculateAccuracy(weight)
            #print("Round {} Accuracy: {:.2%}".format(rd, accuracy))
            
        # stop algo. when an entire check result in zero error
        print("\n<--------------------My Perceptron Algorithm Result-------------------->")
        print("Weight: {}".format(weight))
        print("Accuracy: {:.2%} after {} rounds".format(accuracy, rd))
        self.plot3D(weight)
        return weight
    
    
    def pocket(self):
        if self.algorithm:
            print("Error Message: please reinitialize MyLinearClassification class")
            return
        self.algorithm = "Pocket"
        # step 0: start from initial weight
        pocket_weight = self.initial_weight
        pocket_misclassification = self.EV.countMisclassifiedPoints(pocket_weight)
        
        misclassification_record = list()
        rd = 0
        weight = pocket_weight
        print("\n\n[Pocket Algorithm Running...]")
        while rd < self.max_iteration:
            rd += 1
            for row, actual_classification in zip(self.X, self.y):
                # step 1: find the next mistake of w
                if self.EV.evaluate(row, weight) != actual_classification:
                    # step 2: correct the mistake
                    weight = weight + row * actual_classification
                    break
                    
            # step 3: compare the number of misclassification between pocket_weight and new_weight
            misclassification = self.EV.countMisclassifiedPoints(weight)
            misclassification_record.append(misclassification)
            
            # keep the weight into the pocket if it is better than current pocket weight
            if misclassification < pocket_misclassification:
                pocket_weight, pocket_misclassification = weight, misclassification
        
        # print result and output plot
        print("\n<--------------------My Pocket Algorithm Result-------------------->")
        print("Weight: {}".format(pocket_weight))
        print("Accuracy: {:.2%}".format(self.EV.calculateAccuracy(pocket_weight)))
        self.plot3D(pocket_weight)
        self.plotAccuracy(misclassification_record)
        
        return pocket_weight


    def plot3D(self, weight):
        plt.ion()
        xx, yy = np.meshgrid(range(2), range(2))
        z = (-weight[1] * xx - weight[2] * yy - weight[0]) * 1. /weight[3]
        # plot the decision boundary(hyperplane)
        plt3d = plt.figure().gca(projection='3d')
        plt3d.plot_surface(xx, yy, z, color = 'tomato', alpha = 0.5, zorder = 1)
        ax = plt.gca()
        
        # scattered data points
        data = np.c_[self.X, self.y]
        positive_data = data[data[:, 4]== 1]
        xdata = positive_data[:, 1]
        ydata = positive_data[:, 2]
        zdata = positive_data[:, 3]
        ax.scatter3D(xdata, ydata, zdata, color = 'steelblue', marker = '+', zorder = 2)
        negative_data = data[data[:, 4]== -1]
        xdata = negative_data[:, 1]
        ydata = negative_data[:, 2]
        zdata = negative_data[:, 3]
        ax.scatter3D(xdata, ydata, zdata, color = 'grey', marker = '_', zorder = 2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('{} Algorithm Classification Result'.format(self.algorithm))
        plt.draw()
        plt.pause(0.001)
        plt.show()
        
    
    def plotAccuracy(self, misclassification_record):
        plt.ion()
        plt.figure(figsize=(10,5))
        X = range(len(misclassification_record))
        y = misclassification_record
        plt.plot(X, y, '-k')
        
        # label for data points
        idx = np.argmin(y)
        plt.text(idx+5, y[idx]-2, "round {} with fewest misclassification {}".format(idx, y[idx]))
        plt.scatter(idx, y[idx], color = "tomato", alpha = 0.5)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('iteration')
        plt.ylabel('misclassified count')
        plt.title('Misclassified Points Against the Number of Iterations')

        plt.draw()
        plt.pause(0.001)
        plt.show()
        



class MyLogisticRegression():
    def __init__(self, X, y, learning_rate = 1e-2):
        self.N = X.shape[0]
        x0 = np.ones(self.N)
        self.X = np.c_[x0, X]
        self.y = y
        self.initial_weight = np.zeros(self.X.shape[1])
        #self.initial_weight = initial_weight
        self.learning_rate = learning_rate
        self.iteration = 7000
        self.gradient_record = list()
        self.accuracy_list = list()
        self.EV = Evaluation(self.X, self.y)
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def computeGradient(self, weight):
        summ = 0
        for n in range(self.N):
            summ += self.sigmoid((-1) * np.dot(self.y[n], np.dot(weight, self.X[n].T))) * (-self.y[n]) * self.X[n]
        return 1/self.N * summ

    
    # in a gradient descent fashion
    def updateWeight(self, previous_weight, gradient):
        size_of_gradient = (np.sum(gradient ** 2)) ** (0.5)
        self.gradient_record.append(size_of_gradient)
        
        return previous_weight - (self.learning_rate) * gradient
    
    
    def logisticRegression(self):
        weight = self.initial_weight
        print("\n\n[Logistic Regression Running...]")
        
        for t in range(self.iteration):
            gradient = self.computeGradient(weight)
            weight = self.updateWeight(weight, gradient)
            accuracy = self.EV.calculateAccuracy(weight)
            self.accuracy_list.append(accuracy)
            
        # print result and output plot
        print("\n<--------------------My Logistic Regression Result-------------------->")
        print("Weight: {}".format(weight))
        print("Accuracy: {:.2%}".format(accuracy))
        self.plotRecord(self.gradient_record, "gradient")
        self.plotRecord(self.accuracy_list, "accuracy")
        return weight, self.gradient_record, self.accuracy_list
    
    
    def plotRecord(self, record, label):
        plt.ion()
        plt.figure(figsize=(10,5))
        # plot line
        x = range(len(record))
        y = record
        plt.plot(x, y, '-k')
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('iteration')
        plt.ylabel(label)
        plt.title("Logistic Regression (learning rate = {})".format(self.learning_rate))

        plt.draw()
        plt.pause(0.001)
        plt.show()



class MyLinearRegression():
    def __init__(self, X, y):
        x0 = np.ones(X.shape[0])
        self.X = np.c_[x0, X]
        self.y = y
        
        
    def train(self):
        return np.dot(((np.linalg.inv(np.dot(self.X.T, self.X))).dot(self.X.T)), self.y)
    
    
    
    
if __name__ == "__main__":
    # Input data
    classification_data = np.loadtxt(folder + "/classification.txt", delimiter = ',')
    regression_data = np.loadtxt(folder + "/linear-regression.txt", delimiter = ',')


    # Perceptron Algorithm
    X, y = classification_data[:,:3], classification_data[:,3]
    LR = MyLinearRegression(X, y)
    weight_linear = LR.train()
    #print("Initial Weight for Perceptron:", weight_linear)
    LC = MyLinearClassification(X, y, weight_linear)
    weight_perceptron = LC.perceptron()
    
    from sklearn.linear_model import Perceptron
    lib_per = Perceptron(tol = 1e-3, random_state = 0)
    lib_per.fit(X, y)
    print("\n<--------------------Sklearn Perceptron Algorithm Result-------------------->")
    print("Intercept:", lib_per.intercept_)
    print("Weight:", lib_per.coef_)
    
    
    # Pocket Algorithm
    X, y = classification_data[:,:3], classification_data[:,4]
    LR = MyLinearRegression(X, y)
    weight_linear = LR.train()
    #print("Initial Weight for Pocket:", weight_linear)
    LC = MyLinearClassification(X, y, weight_linear)
    weight_pocket = LC.pocket()
    
    
    # Logistic Regression
    X, y = classification_data[:,:3], classification_data[:,4]
    LogR = MyLogisticRegression(X, y, 0.5)
    weight_logistic5, gradient_record, accuracy_list = LogR.logisticRegression()
    
    from sklearn.linear_model import LogisticRegression
    lib_logi = LogisticRegression(solver = 'sag').fit(X, y)
    print("\n<--------------------Sklearn Logistc Regression Result-------------------->")
    print("Intercept:", lib_logi.intercept_)
    print("Weight:", lib_logi.coef_)
    # Librar Evaluation
    x0 = np.ones(X.shape[0])
    X = np.c_[x0,X]
    EV = Evaluation(X, y)
    library_weight = np.c_[lib_logi.intercept_, lib_logi.coef_].T
    accuracy = EV.calculateAccuracy(library_weight)
    print("Accuracy: {:.2%}".format(accuracy))
    
    
    # Linear Regression
    X, y = regression_data[:, :2], regression_data[:, 2]
    LR = MyLinearRegression(X, y)
    weight_linear = LR.train()
    print("\n<--------------------My Linear Regression Result-------------------->")
    print("Weight: {}".format(weight_linear))
        
    
    from sklearn.linear_model import LinearRegression
    lib_reg = LinearRegression().fit(X, y)
    print("\n<--------------------Sklearn Linear Regression Result-------------------->")
    print("Intercept:", lib_reg.intercept_)
    print("Weight:", lib_reg.coef_)
    
    
    
    # for avoiding the output graph disapear immediately after finishing execution
    print("\n<--------------------Execution End-------------------->")
    input("Press [enter] to close all graph windows.\n")