#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""
import numpy as np
import math
import random as rd

import matplotlib.pyplot as plt



class MyFastMap():
    def __init__(self, input_data, input_words, create_dimension):
        self.input_words = input_words
        self.N = len(self.input_words)
        # build original distance matrix
        self.dist_matrix_org = np.zeros((self.N, self.N), dtype = float)
        for row in input_data:
            o1 = int(row[0]) - 1
            o2 = int(row[1]) - 1
            dist = row[2]
            self.dist_matrix_org[o1][o2] = dist ** 2
            self.dist_matrix_org[o2][o1] = dist ** 2
        
        self.K = create_dimension
        self.find_farthest_iter = 3
        

    """ 
    find two most distance points recursively
    break when previous pivot is identical to the next pivot
    or iteration threshold reached
    """
    def findFarthestPair(self, dist_matrix):
        previous_pivot = int(rd.random() * self.N)
        this_pivot = np.argmax(dist_matrix[previous_pivot])
        next_pivot = np.argmax(dist_matrix[this_pivot])
        
        iterr = 0
        while next_pivot != previous_pivot and iterr < self.find_farthest_iter:
            previous_pivot, this_pivot = this_pivot, next_pivot
            next_pivot = np.argmax(dist_matrix[this_pivot])
        
        return this_pivot, next_pivot
    
    
    """ 
    new distance = previous distance - explained information
    """
    def updateDistanceMatrix(self, dist_matrix, data_points, last_coord):
        for row in range(self.N):
            for col in range(row + 1, self.N):
                info_explained = (data_points[row][last_coord] - data_points[col][last_coord]) ** 2
                new_dist = dist_matrix[row][col] - info_explained
                dist_matrix[row][col] = new_dist
                dist_matrix[col][row] = new_dist
        return dist_matrix
    

    """ 
    project object onto the segment of choosen landmarks
    return the length between projection and landmark1
    """
    def project(self, dist_matrix, object_idx, landmark1, landmark2):
        return (dist_matrix[object_idx][landmark1] + dist_matrix[landmark1][landmark2] - dist_matrix[object_idx][landmark2])/ (math.sqrt(dist_matrix[landmark1][landmark2]) * 2)
    
    
    """
    create one artificial coordinates in each iteration
    """
    def constructCoordinates(self):
        data_points = np.array([]).reshape((self.N, 0))
        iterr = 0
        dist_matrix = self.dist_matrix_org
        while iterr < self.K:
            landmark1, landmark2 = self.findFarthestPair(dist_matrix)
            new_coord = list(map(lambda x: self.project(dist_matrix, x, landmark1, landmark2), range(self.N)))
            # length between projected point and landmark1 is the coordinate
            data_points = np.c_[data_points, new_coord]
            
            if iterr == self.K - 1:
                # no need to update distance matrix
                break
            
            dist_matrix = self.updateDistanceMatrix(dist_matrix, data_points, iterr) # number of iterr = index of last coordinate
            iterr += 1
            print("{} {}D points created for {} objects.".format(self.N, self.K, self.N))
        return data_points
    
    
    def plotWords2D(self, data_points):
        fig = plt.figure()
        x = list(map(lambda x: x[0], data_points))
        y = list(map(lambda x: x[1], data_points))
        plt.scatter(x, y, c = 'steelblue', label = 'artificial points for words')
        # text label beside data points
        for i in range(self.N):
            plt.text(x[i]+0.2, y[i]-0.1, self.input_words[i])
            
        plt.xlim(-0.5, np.max(data_points, axis = 0)[0] + 4)
        plt.ylim(-0.5, np.max(data_points, axis = 0)[1] + 2)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('FastMap - plotting words in 2D')
        plt.show()