#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""

import math

input_txt = "/Users/stanley/Documents/USC/2020_Courses/Spring20_Courses/INF552/DecisionTree/dt_data.txt"
input_rows = 22

"""Returns the distribution of label"""
# I put this function here since i'm using it in different classes
def classCounts(rows):
    counts = {"No": 0, "Yes": 0}  # A dictionary of label (No/ Yes) count
    for row in rows:
        label = row[-1] # In our dataset format, the label is always at the last column
        counts[label] += 1
    return counts

        
"""A Decision Node asks a question.
This holds 
(1) a reference to the question,
(2) and the branches (another Decision Node) after each possible answer.
"""
class DecisionNode():
    def __init__(self, question, categories, branches):
        self.question = question # The column number, not column name, use DT.col_name to retrieve column name
        self.branches = branches # A dict with (key, value) = (ans for question, next Decision Node)


"""A Leaf node classifies data.
This holds a dictionary of label Yes/ No -> number of times
it appears in the rows from the training data that reach this leaf.
"""
class Leaf():
    def __init__(self, rows):
        self.predictions = classCounts(rows)
        

class DecisionTree():
    def __init__(self): 
        # Import data
        f = open(input_txt, "r")
        
        # self.col_names: Column labels
        self.col_names = f.readline()[1: -2].split(", ")
        self.remainingAttrSet = set(range(len(self.col_names) - 1))
        
        # self.train_data: Training dataset
        f.readline()
        self.train_data = []
        for i in range(input_rows):
            data = f.readline()[4:-2].split(", ")
            self.train_data.append(data)
    
    
    """ Split data by unique attribute value """
    def splitData(self, col, data):
        data_subsets = dict()
        for row in data:
            category = row[col]
            # For example, category of Occupied is "High", "Moderate", "Low"
            if category not in data_subsets:
                data_subsets[category] = list()
            data_subsets[category].append(row)
        return data_subsets
    
    
    def getCurrentEntropy(self, data):
        label_dict = classCounts(data)
        if label_dict["No"] * label_dict["Yes"] == 0:
            # Pure state
            return 0
        p1 = label_dict["No"]/sum(label_dict.values()) # Pr of rows with Enjoy == "No"
        p2 = 1 - p1
        currentEntropy = -p1 * math.log2(p1) - p2 * math.log2(p2)
        return currentEntropy
    
      
    def calculateAvgEntropy(self, col_nb, data):
        total_row = len(data)
        data_subsets = self.splitData(col_nb, data)
        avg_entropy = 0
        for category in data_subsets.keys():
            splitted_data = data_subsets[category]
            count_dict = classCounts(splitted_data)
            # Use try in case we have a pure state, 
            # which leads to error in computing entropy 
            try:
                p1 = count_dict["No"]/sum(count_dict.values()) # Pr of rows with Enjoy == "No"
                p2 = 1 - p1
                event_entropy = -p1 * math.log2(p1) - p2 * math.log2(p2)
                avg_entropy += (sum(count_dict.values())/total_row) * event_entropy
            except:
                continue
            
        return avg_entropy
    
    
    """ Find out the most valueable question among subset of data recursively"""
    def findBestGain(self, current_entropy, remainingAttrSet, remainingData):
        best_gain = 0
        picked_col = 0
        for col_nb in remainingAttrSet:
            avg_entropy = self.calculateAvgEntropy(col_nb, remainingData)
            # print("avg_entropy of", self.col_names[col_nb],"is", avg_entropy)
            gain = current_entropy - avg_entropy
            if gain > best_gain:
                best_gain, picked_col = gain, col_nb
        return best_gain, picked_col
    
    
    """ Split data and grow branches according to information gain in a greedy fashion"""
    def buildTree(self, remainingAttrSet, remainingData):
        # Base cases : terminate if pure state/ no more question to ask
        # Pure state
        current_entropy = self.getCurrentEntropy(remainingData)
        if current_entropy == 0:
            return Leaf(remainingData)
        # No question left
        if len(remainingAttrSet) == 0:
            return Leaf(remainingData)
        
        best_gain, picked_col = self.findBestGain(current_entropy, remainingAttrSet, remainingData)
        
        if best_gain == 0:
            return Leaf(remainingData)
        
        # Split data
        data_subsets = self.splitData(picked_col, remainingData)
        # print("picked question and split data by", self.col_names[picked_col])
        
        remainingAttrSet.discard(picked_col)
        
        # Grow branches for a decision node
        branches = dict()
        for category in data_subsets.keys():
            branches[category] = self.buildTree(remainingAttrSet.copy(), data_subsets[category])
        
        return DecisionNode(picked_col, data_subsets.keys(), branches)
    
    
    """Print until leaf of that branch is reached"""
    def print_tree(self, node, branch = "|", indent=""):
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print (indent + "  Predict", node.predictions)
            return
        
        # Print the question at this node
        print (indent + " Q: " + str(self.col_names[node.question]) + "?")
        indent += "  "
        # Call this function recursively on branches
        for category in node.branches.keys():
            print (indent + branch + "-- " + str(category))
            self.print_tree(node.branches[category], branch, indent + "|  ")
        print(indent + "|" + "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
        print(indent)
    
    
    """Take test data and output leaf node"""
    def classify(self, node, datarow):
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions
        column = node.question
        # print("classify reach node", self.col_names[node.question])
        value = datarow[column]
        # print("go to child", value)
        # Decide which children to follow next
        return self.classify(node.branches[value], datarow)
    
    
    """Read in a leaf and print out prediction"""
    def print_leaf(self, counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

if __name__ == '__main__':
    DT = DecisionTree()
    # Loop through each remaining attirbute 
    # to see which one has the least avg. entropy (most information gain)
    my_tree = DT.buildTree(DT.remainingAttrSet, DT.train_data)
    print("------------------------------------")
    print("[My Decision Tree]")
    DT.print_tree(my_tree)
    testing_data = [["Moderate", "Cheap", "Loud", "City-Center", "No", "No"]]
    print("Prediction for Test Data", testing_data)
    for row in testing_data:
        print(DT.print_leaf(DT.classify(my_tree, row)))
    