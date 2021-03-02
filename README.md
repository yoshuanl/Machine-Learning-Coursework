# Machine-Learning-Coursework

This repository contains my implementations of some well-known machine learning algorithms. 
These are individual assignments of the course INF552 - Machine Learning for Data Science taught by professor [Satish Thittamaranahalli Ka](https://www.cs.usc.edu/directory/faculty/profile/?lname=Thittamaranahalli&fname=Satish).

Topics included: 
* Decision Tree
* K-Means
* GMM
* PCA
* FastMap
* Linear Regression
* Back Propagation algorithm for Feed Forward Neural Networks
* SVM
* HMM

Under each topic's folder, you can find:
1. assignment instruction
2. a Python code which I implemented the algorithm myself instead of using library functions
3. training and testing data
4. a report which includes: description of the data structure I used, any code-level optimizations I performed, any challenges I faced, and the prediction result

Below, I give an example of how the report would include, feel free to visit above folders to find out more.

## Hidden Markov Model - Viterbi Algorithm
### Input Data
[hmm-data.txt](7_HMM/hmm-data.txt)

### Data Structure
I store all the input information in a dictionary, with keys: grid_world, tower_loc and footprint. 
Using dictionary allows me to read input file more intelligent (without specifying the actual number of line to read for three different information).

Since the most likely trajectory of the robot is obtained by retracing from the last step, I need to store the optimal path for every step. 
Instead of storing the whole path from the first step, I store only the previous position for each step, and call it as the “back-pointer”. 
In other words, for each step, I use a 2D numpy array to represent the grid world, and for each grid, I store from which grid the robot came from from previous step, that is, which neighboring grid along with this grid forms the highest probability to produce provided observation.

### Result
The most possible trajectory of Robot for 11 time-steps is as follow:

[(5, 3), (6, 3), (7, 3), (8, 3), (8, 2), (7, 2), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1)]

![alt text](https://github.com/yoshuanl/Machine-Learning-Coursework/blob/master/7_HMM/robot_trajectory.png)

### Implementation
Hidden state: the real position of the robot in the grid world
Observation: noisy distances to towers 1, 2, 3 and 4 respectively for 11 time-steps

I calculate the probability of the robot stepping on each grid base on the probability of its free neighboring grid’s on the previous step, the count of free neighbors of this neighboring grid, and the current step’s input distance record. 
Since each grid has four neighbors, I choose the one which produce the largest probability and set a back-pointer pointing back to this neighboring grid. 
After 11 time-steps, I find out the grid with the highest probability for the last step, and mark it as the final grid the robot step on. 
And then backtrack the robot’s trajectory base on the back-pointer I stored for each step.

### Challenges I Face: the Space Complexity
Beside the input data, there are some data generated through the process. At first, I store the probability for each grid in each step. 
This forms a 11x10x10 matrix after all calculations. I also store the “back-pointer” for each grid in each step. 
This forms another 11x10x10 matrix after all calculations. 
Although these doesn’t exceed my computer’s capacity, I believe I could further improve the performance of my implementation by reducing the space complexity.

### Optimization
I found out that storing the probability for each grid in each step is unnecessary. 
After all, we only need the probabilities of standing on each grid from the last step (11th time-step), and find out which grid in the last step has the largest probability. 
Therefore, I store only the probabilities of each grid for current and previous step only.

Moreover, I write my code in an object oriented manner. 
I also put parameters such as the noise interval coefficient at the very front of my class to avoid any magic numbers in the code.
