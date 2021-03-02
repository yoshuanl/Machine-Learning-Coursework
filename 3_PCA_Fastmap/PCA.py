#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA



class MyPCA():
    def __init__(self, input_data, project_to_dimension):
        self.input_data = input_data
        self.N = len(input_data)
        self.K = project_to_dimension


    def fitPca(self):
        # centralize each column
        column_mu = np.mean(self.input_data, axis = 0)
        centralized_input = self.input_data - column_mu
        
        t_centralized_input = np.transpose(centralized_input)
        cov_mat = t_centralized_input.dot(centralized_input) / (self.N)
        
        # eigenvectors and eigenvalues from the covariance matrix
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        eig_pairs = [(eig_val[i], eig_vec[i]) for i in range(len(eig_val))]
        
        # pick eigenvectors with K highest eigenvalues
        eig_pairs.sort(key = lambda x: -x[0])
        picked_eig_pairs = eig_pairs[: self.K]
        picked_eig_vec = np.asarray(list(map(lambda x: x[1], picked_eig_pairs))).T
        picked_eig_val = list(map(lambda x: x[0], picked_eig_pairs))
        
        picked_eig_val_sum = np.sum(picked_eig_val)
        eig_val_sum = np.sum(list(map(lambda x: x[0], eig_pairs)))
        variance_explained = picked_eig_val_sum / eig_val_sum
        
        # linear transformation
        my_transformation = centralized_input.dot(picked_eig_vec)
        
        return my_transformation, picked_eig_vec, picked_eig_val, variance_explained


    def compareSkLib(self, my_transformation):
        sklearn_pca = sklearnPCA(n_components = self.K)
        sklearn_transformation = sklearn_pca.fit_transform(self.input_data)
        
        sklearn_transformation = sklearn_transformation * (-1) # eigenvector can be either positive/ negative direction
        print("Variance explained by sklearn PCA: {:.2%}".format(sum(sklearn_pca.explained_variance_ratio_)))
        if my_transformation.all() == sklearn_transformation.all():
            print("\nIdentical Transformation Result with Sklearn Library!!")
           

    def printResult(self, my_transformation, picked_eig_vec, variance_explained):
        print("\nDirections of the principal components:")
        for i in range(self.K):
            print("({})  {}".format(i + 1, picked_eig_vec.T[i]))
        
        print("\nVariance explained by my PCA: {:.2%}".format(variance_explained))
        
        self.compareSkLib(my_transformation)
    
    
    def plot3D(self, picked_eig_vec, picked_eig_val):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # scattered data points
        xdata = self.input_data[:, 0]
        ydata = self.input_data[:, 1]
        zdata = self.input_data[:, 2]
        ax.scatter3D(xdata, ydata, zdata, color = 'steelblue', alpha = 0.05, zorder = 1)
        # eigenvectors
        v = picked_eig_vec.T
        origin = [0], [0], [0]
        color = ['tomato','tomato']
        for i in range(len(picked_eig_val)):
            ax.quiver(*origin, v[i, 0], v[i, 1], v[i, 2], color = color[i], length = picked_eig_val[i], zorder = 2)
        
        ax.set_xlim(-50,50)
        ax.set_xlim(-25,25)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('PCA - plotting original 3D data points')
        plt.show()
        
        
    def plot2D(self, my_transformation):
        fig = plt.figure()
        x = list(my_transformation[:, 0])
        y = list(my_transformation[:, 1])
        plt.scatter(x, y, c = 'black', alpha = 0.1)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('pca1')
        plt.ylabel('pca2')
        plt.title('PCA - plotting transformed 2D data points')
        plt.show()
        

