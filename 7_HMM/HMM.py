#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""
import sys
import os
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

wkdir = folder = sys.path[0] + "/"
#wkdir = "/Users/stanley/Documents/USC/Courses/Spring20_Courses/INF552/7_HMM/"
os.chdir(wkdir)


class HMM():
    def __init__(self, input_data):
        self.input_data = input_data
        self.row_cnt, self.col_cnt = input_data["grid_world"].shape
        self.total_step = len(input_data["footprint"])
        
        self.multiple_decimal_to_integer = 10
        self.noise_multiplier_lower_limit = 0.7 * self.multiple_decimal_to_integer
        self.noise_multiplier_upper_limit = 1.3 * self.multiple_decimal_to_integer
        
        self.distance_matrix = [[] for _ in range(self.row_cnt)]
        self.neighbor_matrix = [[] for _ in range(self.row_cnt)]
        
    
    """ Calculate the euclidean distance to towers """
    def calculateDistanceToTower(self, x, y):
        dist_to_towers = list()
        for tower in self.input_data["tower_loc"]:
            tower_x, tower_y = tower
            d = ((x - tower_x) ** 2 + (y - tower_y) ** 2) ** (1/2)
            dist_to_towers.append(d)
        return dist_to_towers


    """ Output the lower and upper limit of the noise interval """
    def noiseRange(self, distance):
        start, end = math.ceil(distance * self.noise_multiplier_lower_limit), math.floor(distance * self.noise_multiplier_upper_limit)
        return start, end


    """ Calculate eulcidean distance from each point to four towers """
    def buildDistanceMatrix(self):
        for row in range(self.row_cnt):
            for col in range(self.col_cnt):
                self.distance_matrix[row].append(self.calculateDistanceToTower(row, col))


    """" Calculate the possibility of having that particular pair of distance record """
    def calculatePossibility(self, step, row, col):
        possibility = 1
        distances = self.distance_matrix[row][col]
        for idx in range(4):
            dist = distances[idx]
            start, end = self.noiseRange(dist)
            footprint = self.input_data["footprint"][step][idx]
            if start <= footprint * self.multiple_decimal_to_integer <= end:
                possibility *= 1/(end - start + 1)
            else:
                possibility = 0
                break
        return possibility


    """ Count neighboring cells != "0" """
    def countAvailableNeighbor(self, row, col):
        neighbor_cnt = 0
        if row - 1 >= 0 and self.input_data["grid_world"][row-1][col] != 0:
            neighbor_cnt += 1
        if row + 1 < self.row_cnt and self.input_data["grid_world"][row+1][col] != 0:
            neighbor_cnt += 1
        if col - 1 >= 0 and self.input_data["grid_world"][row][col-1] != 0:
            neighbor_cnt += 1
        if col + 1 < self.col_cnt and self.input_data["grid_world"][row][col+1] != 0:
            neighbor_cnt += 1
        return neighbor_cnt
    
    
    def buildNeighborMatrix(self):
        for row in range(self.row_cnt):
            for col in range(self.col_cnt):
                self.neighbor_matrix[row].append(self.countAvailableNeighbor(row, col))


    def viterbi(self):
        prev_dp = np.zeros(input_data["grid_world"].shape)
        backpointer_list = list()
        # initialize dp
        for row in range(self.row_cnt):
            for col in range(self.col_cnt):
                if input_data["grid_world"][row][col] == 0:
                    continue
                possibility = self.calculatePossibility(0, row, col)
                prev_dp[row][col] = possibility
        # dp
        for step in range(1, self.total_step):
            dp = np.zeros(self.input_data["grid_world"].shape)
            backpointer = np.zeros(self.input_data["grid_world"].shape)
            for row in range(self.row_cnt):
                for col in range(self.col_cnt):
                    if self.input_data["grid_world"][row][col] == 0:
                        continue
                    possibility = self.calculatePossibility(step, row, col)
                    if possibility == 0:
                        continue
                    maxx = 0
                    if row - 1 >= 0 and self.input_data["grid_world"][row-1][col] != 0:
                        candidate = prev_dp[row-1][col] / self.neighbor_matrix[row-1][col]
                        if candidate > maxx:
                            maxx = candidate
                            pointer = (row-1) * self.row_cnt + col
                    if row + 1 < self.row_cnt and self.input_data["grid_world"][row+1][col] != 0:
                        candidate = prev_dp[row+1][col] / self.neighbor_matrix[row+1][col]
                        if candidate > maxx:
                            maxx = candidate
                            pointer = (row+1) * self.row_cnt + col
                    if col - 1 >= 0 and self.input_data["grid_world"][row][col-1] != 0:
                        candidate = prev_dp[row][col-1] / self.neighbor_matrix[row][col-1]
                        if candidate > maxx:
                            maxx = candidate
                            pointer = row * self.row_cnt + col-1
                    if col + 1 < self.col_cnt and self.input_data["grid_world"][row][col+1] != 0:
                        candidate = prev_dp[row][col+1] / self.neighbor_matrix[row][col+1]
                        if candidate > maxx:
                            maxx = candidate
                            pointer = row * self.row_cnt + col+1
                    dp[row][col] = maxx * possibility
                    backpointer[row][col] = pointer
            
            prev_dp = dp.copy()
            backpointer_list.append(backpointer)
        
        # backtrack footprint
        footprint = list()
        location = np.argmax(dp)
        grid_x, grid_y = int(location // self.col_cnt), int(location % self.col_cnt)
        footprint.append((grid_x, grid_y))
        step = self.total_step - 2
        while step >= 0:
            location = backpointer_list[step][grid_x][grid_y]
            grid_x, grid_y = int(location // self.col_cnt), int(location % self.col_cnt)
            footprint.append((grid_x, grid_y))
            step -= 1
            
        return footprint[::-1]


    def plotFootprint(self, footprint):
        plt.ion()
        plt.figure()
        x = list(map(lambda z: z[1], footprint))
        y = list(map(lambda z: z[0], footprint))
        plt.plot(x, y, '-o', color='black')
        plt.plot(x[0], y[0], 'v', markersize=13, color='steelblue', label = "start")
        plt.plot(x[-1], y[-1], '^', markersize=13, color='lightcoral', label = "end")
        plt.xlim(0, self.col_cnt - 1)
        plt.ylim(0, self.row_cnt - 1)
        plt.legend()
        plt.title("The Possible Trajectory of Robot")
        
        ax = plt.gca()                            # get the axis
        ax.set_ylim(ax.get_ylim()[::-1])          # invert the axis
        ax.xaxis.tick_top()                       # and move the X-Axis  
        ax.yaxis.tick_left()                      # remove right y-Ticks



# for cleaning distance record
def cleanDistRecord(x):
    if x[2] == "":
        return list(map(float, x[:2] + x[3:]))
    else:
        return list(map(float, x))
    
    
if __name__ == "__main__":
    
    """ Data input/ cleansing"""
    input_data = collections.defaultdict(list)
    with open("hmm-data.txt", "r") as f:
        for line in f:
            row = line[:-1].split(" ")
            if len(row) == 1 or row[-1][-1] == ":":
                if row[0] == "":
                    continue
                if row[0].find("Grid") >= 0:
                    curr = "grid_world"
                elif row[0].find("Tower") >= 0:
                    curr = "tower_loc"
                else:
                    curr = "footprint"
            else:
                input_data[curr].append(row)
                
    input_data["grid_world"] = np.array(input_data["grid_world"], dtype = int)
    input_data["tower_loc"] = list(map(lambda x: (int(x[2]), int(x[3])), input_data["tower_loc"]))
    input_data["footprint"] = list(map(cleanDistRecord, input_data["footprint"]))
    
    """ HMM Viterbi algorithm """
    print("\n<--------------------HMM Algorithm-------------------->")
    MyHMM = HMM(input_data)
    MyHMM.buildDistanceMatrix()
    MyHMM.buildNeighborMatrix()
    nb=MyHMM.neighbor_matrix
    footprint = MyHMM.viterbi()
    print("Trajectory of Robot:\n", footprint)
    MyHMM.plotFootprint(footprint)
    
    # for avoiding the output graph disapear immediately after finishing execution
    print("\n<--------------------Execution End-------------------->")
    input("Press [enter] to close all graph windows.\n")