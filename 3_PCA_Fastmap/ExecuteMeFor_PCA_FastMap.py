#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""
"""
This file imports PCA.py and FastMap.py as module.
Please modify file path or parameters in this file if needed.
"""
# for enabling interactive 3D plot, please execute below line before plotting
# %matplotlib auto

import os
import numpy as np

# data file path and parameters
#folder = os.path.dirname(os.path.realpath(__file__))
folder = "/Users/stanley/Documents/USC/Courses/Spring20_Courses/INF552/PCA_Fastmap"
PCA_project_to_dimension = 2
FM_create_dimension = 2

os.chdir(folder)
import FastMap
import PCA

# Inputs
data_pca = np.loadtxt(folder + "/pca-data.txt", delimiter = '\t')
data_fastmap = np.loadtxt(folder + "/fastmap-data.txt", delimiter = '\t')
with open(folder + "/fastmap-wordlist.txt") as f:
    word_label_fastmap= [line.rstrip() for line in f]


# Alogorithm Execution
print("\n<--------------------PCA-------------------->")
My_PCA = PCA.MyPCA(data_pca, PCA_project_to_dimension)
my_transformation, picked_eig_vec, picked_eig_val, variance_explained = My_PCA.fitPca()
My_PCA.plot3D(picked_eig_vec, picked_eig_val)

My_PCA.plot2D(my_transformation)
My_PCA.printResult(my_transformation, picked_eig_vec, variance_explained)

print("\n<--------------------FastMap-------------------->")
FP = FastMap.MyFastMap(data_fastmap, word_label_fastmap, FM_create_dimension)
data_points = FP.constructCoordinates()
FP.plotWords2D(data_points)