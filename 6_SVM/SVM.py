#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""

# If the script is run in python IDE,
# for enabling interactive 3D plot, please execute below line before plotting
# %matplotlib auto
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import svm

from quadprog import solve_qp
# https://github.com/rmcgibbo/quadprog/blob/master/quadprog/quadprog.pyx
# Minimize     1/2 x^T G x - a^T x
# Subject to   C.T x >= b

#folder = sys.path[0] + "/"
folder = "/Users/stanley/Documents/USC/Courses/Spring20_Courses/INF552/6_SVM"
class SVM():
    def __init__(self, input_data):
        self.input_data = input_data
        self.X = input_data[:, :2]
        self.y = input_data[:, 2]
        self.total_input = input_data.shape[0]
        self.tolerance = 1e-10
    
    
    def kernelTrick(self, kernel, x, y):
        if kernel == "Gaussian":
            return self.gaussianKernel(x, y)
        else:
            return self.polynomialKernel(x, y)
        
    
    def polynomialKernel(self, x, y, dimension = 2):
        return (np.dot(x, y.T)) ** dimension


    def gaussianKernel(self, x, y, gamma = 0.2):
        sub = (x - y) ** 2
        if len(sub.shape) < 2:
            return np.exp(-gamma * np.sum(sub))
        return np.exp(-gamma * np.sum(sub, axis = 1))
    
    
    def solvePrimalParameters(self):
        G = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) + np.eye(3) * 1e-3
        a = np.array([[0., 0., 0.]]).reshape((3, ))
        inequality_C = np.c_[np.ones((100, 1)), self.X] * self.y.reshape(100, 1)
        ineq_b = np.ones((100, ))
        weights = solve_qp(G, a, inequality_C.T, ineq_b)[0]
        
        value = np.dot(weights, np.c_[np.ones((100,1)), self.X].T)
        sv_index = list()
        for idx in range(self.total_input):
            if abs(value[idx]) - 1 < self.tolerance:
                sv_index.append(idx)
        bias, w1, w2 = weights
        return bias, w1, w2, sv_index, value
    
    
    def solveDualParameters(self, kernel):
        G = np.zeros((self.total_input, self.total_input))
        for i in range(self.total_input):
            for j in range(self.total_input):
                dot = self.y[i] * self.y[j] * self.kernelTrick(kernel, self.X[i], self.X[j])
                G[i][j], G[j][i] = dot, dot
        # make it a positive definite matrix
        if kernel == "Polynomial":
            G += np.eye(self.total_input) * 1e-3
            
        a = np.ones((self.total_input, ))
        
        equality_C = self.y
        eq_b = np.array([0.])
        
        inequality_C = np.eye(self.total_input)
        ineq_b = np.zeros((self.total_input, ))
        
        qp_C = np.vstack([equality_C, inequality_C]).T
        qp_b = np.hstack([eq_b, ineq_b])
        meq = 1
        alpha = solve_qp(G, a, qp_C, qp_b, meq)[0]
        
        return alpha
    
    
    def calculateParameters(self, kernel, alpha):
        sv_index = list()
        sv_alpha = list()
        sv_y = list()
        sv_X = np.array([]).reshape(0, 2)
        for idx in range(self.total_input):
            if alpha[idx] > self.tolerance:
                sv_index.append(idx)
                sv_alpha.append(alpha[idx])
                sv_y.append(self.y[idx])
                sv_X = np.r_[sv_X, self.X[idx].reshape(1,2)]
        sv_alpha = np.array(sv_alpha)
        sv_y = np.array(sv_y)
        
        # calculate bias term
        bias = 0
        for idx in sv_index:
            bias += self.y[idx] - sum(sv_alpha * sv_y * self.kernelTrick(kernel, self.X[idx], sv_X))
        bias /= len(sv_index)
        return sv_index, sv_alpha, sv_X, sv_y, bias
    
    
    def predict(self, kernel, test_X, sv_index, sv_alpha, sv_X, sv_y, bias):
        result = list()
        for t in test_X:
            if sum(self.kernelTrick(kernel, t, sv_X) * sv_alpha * sv_y) + bias > 0:
                result.append(1)
            else:
                result.append(-1)
        return np.asarray(result)
    
    
    def plotData(self, sv_index = None):
        plt.ion()
        plt.figure()
        # plot data points
        positive_data = self.input_data[self.input_data[:, 2]== 1]
        x1 = positive_data[:, 0]
        x2 = positive_data[:, 1]
        plt.scatter(x1, x2, color = 'lightcoral', label = "positive")
        negative_data = self.input_data[self.input_data[:, 2]== -1]
        x1 = negative_data[:, 0]
        x2 = negative_data[:, 1]
        plt.scatter(x1, x2, color = 'steelblue', label = "negative")
        plt.legend()
        if sv_index:
            # plot SVs
            SV_x1 = list(map(lambda x: self.X[x][0], sv_index))
            SV_x2 = list(map(lambda x: self.X[x][1], sv_index))
            plt.plot(SV_x1, SV_x2, 'x', color = 'red', markersize = 8, label = "Support Vectors")
            plt.legend(loc = 'upper right')
            
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    
    def plotlinearKernel(self, bias, w1, w2):
        x = np.linspace(0, 0.6, 3)
        y = (w1 * x + bias) / (-w2)
        y_pos = (w1 * x + bias - 1) / (-w2)
        y_neg = (w1 * x + bias + 1) / (-w2)
        plt.plot(x, y, color = "black")
        plt.plot(x, y_pos, "--", color = "grey", alpha = 0.5)
        plt.plot(x, y_neg, "--", color = "grey", alpha = 0.5)
        
        plt.legend()
        plt.title('My Linear SVM')
        plt.show()
    
    
    def plotNonlinearKernel(self, kernel, sv_index, sv_alpha, sv_X, sv_y, bias):
        h = 0.08
        # create a mesh to plot in
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = self.predict(kernel, np.c_[xx.ravel(), yy.ravel()], sv_index, sv_alpha, sv_X, sv_y, bias)
        Z = Z.reshape(xx.shape)
        
        cs = plt.contour(xx, yy, Z, cmap = plt.cm.Paired)
        cs.collections[0].set_label("{} Boundary".format(kernel))
        plt.legend(loc = 'upper right')
        plt.title('My {} SVM'.format(kernel))
        plt.show()
    
    
    def solveKernel(self, kernel):
        if kernel == "Linear":
            bias, w1, w2, sv_index, value = self.solvePrimalParameters()
            self.plotData(sv_index)
            self.plotlinearKernel(bias, w1, w2)
            return sv_index, bias, w1, w2
        else:
            alpha = self.solveDualParameters(kernel)
            sv_index, sv_alpha, sv_X, sv_y, bias = self.calculateParameters(kernel, alpha)
            if kernel == "Polynomial":
                self.plotData(sv_index)
            else:
                self.plotData(sv_index = None)
            self.plotNonlinearKernel(kernel, sv_index, sv_alpha, sv_X, sv_y, bias)
            return sv_index
        

if __name__ == "__main__":
    # Linearly Seperable Data
    input_txt = folder + "/linsep.txt"
    input_data_lin = np.loadtxt(input_txt, delimiter = ',')
    MySVM = SVM(input_data_lin)
    print("\n<--------------------Linear SVM-------------------->")
    #MySVM.plotData(sv_index = None)
    sv_index_lin, bias, w1, w2 = MySVM.solveKernel("Linear")
    print("Used {} Support Vectors".format(len(sv_index_lin)))
    print("Support Vectors are Data Points:")
    for idx in sv_index_lin:
        print("{}th: {}".format(idx, input_data_lin[idx][:2]))
    print("Intercept: {}, \nWeights: {}".format(bias, [w1, w2]))
    
            
    # Linearly Inseperable Data
    input_txt = folder + "/nonlinsep.txt"
    input_data_nonlin = np.loadtxt(input_txt, delimiter = ',')
    MySVM = SVM(input_data_nonlin)
    print("\n<--------------------Polynomial SVM-------------------->")
    #MySVM.plotData(sv_index = None)
    sv_index_poly = MySVM.solveKernel("Polynomial")
    print("Used {} Support Vectors".format(len(sv_index_poly)))
    print("Support Vectors are Data Points:")
    for idx in sv_index_poly:
        print("{}th: {}".format(idx, input_data_nonlin[idx][:2]))
    print("\n<--------------------Gaussian SVM-------------------->")
    sv_index_gauss = MySVM.solveKernel("Gaussian")
    print("Used {} Support Vectors".format(len(sv_index_gauss)))
    print("Support Vectors are Data Points (indices):", sv_index_gauss)
    
    
    # sklearn
    # linear
    clf = svm.LinearSVC()
    clf.fit(input_data_lin[:, :2], input_data_lin[:, 2])
    bias = clf.intercept_[0]
    w1, w2 = clf.coef_[0]
    xx = np.linspace(0, 0.6, 3)
    yy = (w1 * xx) / (-w2)
    MySVM = SVM(input_data_lin)
    MySVM.plotData(sv_index = None)
    plt.plot(xx, yy, color = "black")
    plt.title('Sklearn Linear SVM')
    plt.show()
    print("Support Vectors are Data Points (indices):", clf.support_)
    
    # gaussian
    clf = svm.SVC(kernel = 'rbf', gamma = 0.2)
    X, y = input_data_nonlin[:, :2], input_data_nonlin[:, 2]
    clf.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.08), np.arange(y_min, y_max, 0.08))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    MySVM = SVM(input_data_nonlin)
    MySVM.plotData(sv_index = None)
    cs = plt.contour(xx, yy, Z, colors = 'red')
    cs.collections[0].set_label('Gaussian (gamma = 0.2)')
    print("Gaussian Support Vectors are Data Points (indices):", clf.support_)
    # poly2
    clf = svm.SVC(kernel = 'poly', degree = 2, gamma = 1)
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contour(xx, yy, Z, colors = 'green')
    cs.collections[0].set_label('Polynomial (degree = 2)')
    """
    # poly3
    clf = svm.SVC(kernel = 'poly', degree = 3)
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contour(xx, yy, Z, colors = 'blue')
    cs.collections[0].set_label('Polynomial (degree = 3)')
    """
    plt.legend(loc = 'upper right')
    plt.title('Sklearn Nonlinear SVMs')
    plt.show()
    print("Polynomial Support Vectors are Data Points (indices):", clf.support_)
    
        
    # for avoiding the output graph disapear immediately after finishing execution
    print("\n<--------------------Execution End-------------------->")
    input("Press [enter] to close all graph windows.\n")
