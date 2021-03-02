#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Member: Yo Shuan Liu (4472-6221-33)
"""

"""
In this code, we are going to implement K Means and GMM algorithm
(1) plot the original data
(2) repeat K-means to get different results (caused by different initial centroids)
(3) choose a result with least WCSS, Within-Cluster Sums of Squares
(4) plot the choosen result, and print out the corresponding centroids

(5) run GMM algorithm using the centroid we just obtained from K Means as initial centroid
(6) plot the final result by assigning each data point to the cluster which has the highest probability (gamma)
"""


import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# Compare Answer with sklearn package
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# input data for KMeans and GMM
input_txt = "/Users/stanley/Documents/USC/Courses/Spring20_Courses/INF552/KMeans_GMM/clusters.txt"
input_data = np.loadtxt(input_txt, delimiter = ',')
total_cluster = 3
reinitialize_centroid = 20
kmeans_maxiteration = 30
kmeans_epsilon = 0.0001

gmm_epoch = 100
gmm_epsilon = 0.0001


"""For visualize original data and clustering result"""
class VisualizationCluster():
    def __init__(self, original_data):
        self.original_data = original_data
        self.total_cluster = total_cluster
        
        # for visualization
        self.cluster_color = ['steelblue','olivedrab','peru']
        self.cluster_labels = ['cluster1','cluster2','cluster3']
        
        
    def plotInputData(self):
        x = list(map(lambda x: x[0], self.original_data))
        y = list(map(lambda x: x[1], self.original_data))
        plt.scatter(x, y, c = 'black', label = 'unclustered data')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Plot of Unclustered Data Points')
        plt.show()
    
    
    """This function is for both KMeans and GMM"""
    def plotClusteredData(self, clustered_data, centroids):
        # plot data points cluster by cluster
        for k in range(self.total_cluster):
            indices = clustered_data[k]
            data = list(map(lambda x: self.original_data[x], indices))
            x = list(map(lambda x: x[0], data))
            y = list(map(lambda x: x[1], data))
            plt.scatter(x, y, c = self.cluster_color[k], label = self.cluster_labels[k])
        # plot centroids
        for k in range(self.total_cluster):
            x = centroids[:, k][0]
            y = centroids[:, k][1]
            plt.scatter(x, y, c = 'red', label = "centroid", marker = "+")
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Plot of Clustered Data Points')
        plt.show()
    
    
    """This function is for GMM"""
    def plotLogLikelihood(self, likelihoods):
        plt.figure(figsize=(10, 10))
        plt.title('Log-Likelihood')
        plt.plot(np.arange(1, gmm_epoch + 1), likelihoods)
        plt.show()


class MyKMeans():
    def __init__(self, input_data, k = total_cluster, max_iter = kmeans_maxiteration, epsilon = kmeans_epsilon):
        self.input_data = input_data
        self.total_input = input_data.shape[0]
        self.total_attributes = input_data.shape[1]
        self.total_cluster = k
        self.max_iter = max_iter
        self.epsilon = epsilon


    def chooseRandomCentroid(self):
        centroids = np.array([]).reshape(self.total_attributes, 0)
        idx = np.random.randint(0, self.total_input, size = self.total_cluster)
        
        for k in range(self.total_cluster):
            centroids = np.c_[centroids, self.input_data[idx[k]]]
        return centroids
    
    
    # update a n * k distance matrix
    def calculateEuclidianDistance(self, centroids):
        distance_matrix = np.array([]).reshape(self.total_input, 0)
        
        for k in range(self.total_cluster):
            dist_with_cent = np.sum((self.input_data - centroids[:, k]) ** 2, axis = 1)
            distance_matrix = np.c_[distance_matrix, dist_with_cent]
        return distance_matrix
    
    
    def transformClusterArray(self, cluster_array):
        cluster_list = [[] for _ in range(self.total_cluster)]
        for data_index in range(self.total_input):
            cluster = cluster_array[data_index]
            cluster_list[cluster].append(data_index)
        return cluster_list
    
    
    def calculateWCSS(self, clustering_result, final_centroids):
        WCSS = 0
        for k in range(self.total_cluster):
            centroid = final_centroids[:, k]
            #data_in_cluster = np.asarray(clustering_result[k])
            data_in_cluster = np.asarray(list(map(lambda x: input_data[x], clustering_result[k])))
            if len(data_in_cluster) == 0:
                continue
            WCSS += np.sum((data_in_cluster - centroid) ** 2)
        
        return WCSS
    
        
    def trainKM(self):
        # random sample data as initial centroid
        centroids = self.chooseRandomCentroid()
        iterr = 0
        this_WCSS = 0
        while iterr <= 1 or iterr < self.max_iter and previous_WCSS - this_WCSS > self.epsilon:
            # calculate distance between points and clusters
            distances = self.calculateEuclidianDistance(centroids)
            
            # reassign cluster in each iteration
            cluster_array = np.argmin(distances, axis = 1) # an 1 * n array
            temp_cluster = self.transformClusterArray(cluster_array)
            
            # recalculate the centroids by cluster
            for k in range(self.total_cluster):
                #centroids[:, k] = np.mean(temp_cluster[k], axis = 0)
                centroids[:, k] = np.mean(np.asarray(list(map(lambda x: self.input_data[x], temp_cluster[k]))), axis = 0)
                
            # convergance criteria
            previous_WCSS = this_WCSS
            this_WCSS = self.calculateWCSS(temp_cluster, centroids)
            iterr += 1

        return temp_cluster, centroids
    
            
    def sklearnOutput(self):
        km_sklearn = KMeans(n_clusters = total_cluster, n_init = reinitialize_centroid, max_iter = self.max_iter, tol = kmeans_epsilon).fit(self.input_data)
        predict_labels = km_sklearn.predict(self.input_data)
        predict_labels = self.transformClusterArray(predict_labels)
        return predict_labels, km_sklearn.cluster_centers_
    
    
    def main(self):
        cluster_result_pool = list()
        WCSS_pool = list()
        VC = VisualizationCluster(input_data)
        
        for different_initial in range(reinitialize_centroid):
            clustering_result, centroids = self.trainKM()
            WCSS = self.calculateWCSS(clustering_result, centroids)
            
            cluster_result_pool.append([clustering_result, centroids])
            WCSS_pool.append(WCSS)
            VC.plotClusteredData(clustering_result, centroids)
            print("Round {} with WCSS = {}".format(different_initial, WCSS))
            
        # pick the result with the smallest WCSS
        result_id = np.argmin(WCSS_pool, axis = 0)
        clustering_result, centroids = cluster_result_pool[result_id]
        sk_clustering_result, sk_centroids = self.sklearnOutput()
        print("Picked result with WCSS = {}".format(WCSS_pool[result_id]))
        
        print("\n--------------------K Means--------------------\n")
        print('Centroids by my implementation:\n', np.asarray(centroids).T) 
        print('Centroids by sklearn:\n', sk_centroids)
        print("\nVisualize my K Means implementation:")
        VC.plotClusteredData(clustering_result, centroids)
        print("\nVisualize sklearn K Means implementation:")
        VC.plotClusteredData(sk_clustering_result, sk_centroids.T)
        return centroids
    

class MyGMM():
    def __init__(self, input_data, centroid_from_KM, k = total_cluster, epoch = gmm_epoch, epsilon = gmm_epsilon):
        self.input_data = input_data
        self.total_input = input_data.shape[0]
        self.total_attributes = input_data.shape[1]
        self.centroid_from_KM = centroid_from_KM
        self.total_cluster = k
        self.epoch = epoch
        self.epsilon = epsilon
        
    
    def gaussian(self, mu, cov):
        return multivariate_normal.pdf(self.input_data, mu, cov).reshape(-1, 1)
    

    def initializeClusters(self):
        clusters = []
        
        # use the KMeans centroids to initialize the GMM
        mu_k = self.centroid_from_KM
        
        for k in range(self.total_cluster):
            clusters.append({
                'pi_k': 1.0 / self.total_cluster,
                'mu_k': mu_k[:, k],
                'cov_k': np.identity(self.total_attributes, dtype = np.float64)
            })
        return clusters
    
    
    # calculate gamma for each point in each cluster
    def expectationStep(self, clusters):
        gamma_denominator = np.zeros((self.total_input, 1), dtype = np.float64)
        
        for cluster in clusters:
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']
            gauss_p = self.gaussian(mu_k, cov_k)
            gamma_numerator = (pi_k * gauss_p).astype(np.float64)
            # sum of all gamma's for one data points
            for i in range(self.total_input):
                gamma_denominator[i] += gamma_numerator[i]
            
            cluster['gamma_numerator'] = gamma_numerator
            
        for cluster in clusters:
            cluster['gamma_denominator'] = gamma_denominator
            cluster['gamma_nk'] = cluster['gamma_numerator'] / gamma_denominator

    # use sum of gamma in one cluster to 
    def maximizationStep(self, clusters):
        for cluster in clusters:
            gamma_nk = cluster['gamma_nk']
            cov_k = np.zeros((self.total_attributes, self.total_attributes))
            
            # sum of all gamma's in one cluster
            N_k = np.sum(gamma_nk, axis = 0)
            
            pi_k = N_k / self.total_input
            mu_k = np.sum(gamma_nk * self.input_data, axis = 0) / N_k
            
            for j in range(self.total_input):
                diff = (self.input_data[j] - mu_k).reshape(-1, 1) # m * 1 array
                cov_k += gamma_nk[j] * np.dot(diff, diff.T) # m * m array
                
            cov_k /= N_k
            
            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k

    
    def getLikelihood(self, clusters):
        sample_likelihoods = np.log(np.array([cluster['gamma_denominator'] for cluster in clusters]))
        return np.sum(sample_likelihoods), sample_likelihoods
    
    
    def trainGmm(self):
        clusters = self.initializeClusters() # cluster info, such as mean, covariance
        likelihoods = np.zeros((self.epoch, )) # log-likelihood of each epoch
        scores = np.zeros((self.total_input, self.total_cluster)) # for each data point, the cluster with highest score is where it belongs
    
        for i in range(self.epoch):
            self.expectationStep(clusters)
            self.maximizationStep(clusters)
    
            likelihood, sample_likelihoods = self.getLikelihood(clusters)
            likelihoods[i] = likelihood
            
            if i > 1 and likelihoods[i] - likelihoods[i - 1] < self.epsilon:
                break
    
            #print('Epoch: {}, Likelihood = {}'.format(i + 1, likelihood))
            
        for i, cluster in enumerate(clusters):
            scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
            
        return clusters, likelihoods, scores, sample_likelihoods
    
    
    def sklearnOutput(self):
        gmm_sklearn = GaussianMixture(n_components = total_cluster, max_iter = gmm_epoch, tol = gmm_epsilon).fit(self.input_data)
        predict_labels = gmm_sklearn.predict(self.input_data)
        predict_labels = self.transformClusterArray(predict_labels)
        return predict_labels, gmm_sklearn.means_, gmm_sklearn.weights_, gmm_sklearn.covariances_
    
    
    def transformClusterArray(self, cluster_array):
        cluster_list = [[] for _ in range(self.total_cluster)]
        for data_index in range(self.total_input):
            cluster = cluster_array[data_index]
            cluster_list[cluster].append(data_index)
        return cluster_list
    
    
    def main(self):
        # my implementation
        clusters, likelihoods, scores, sample_likelihoods = self.trainGmm()
        mean = [cluster['mu_k'].tolist() for cluster in clusters]
        amplitude = [cluster['pi_k'].tolist() for cluster in clusters]
        covariance = [cluster['cov_k'].tolist() for cluster in clusters]
        # label data according to scores
        cluster_array = np.argmax(scores, axis = 1)
        clustering_result = self.transformClusterArray(cluster_array)
        
        # sklearn result
        sk_clustering_result, sk_mean, sk_amplitude, sk_cov = self.sklearnOutput()
        
        # print results
        print("\n--------------------GMM--------------------\n")
        print('GMM Means by our implementation:\n', np.asarray(mean))
        print('GMM Means by sklearn:\n', sk_mean)
        
        print('GMM Amplitude by our implementation:\n', np.asarray(amplitude))
        print('GMM Amplitude by sklearn:\n', sk_amplitude)
        
        print('GMM Covariance by our implementation:\n', np.asarray(covariance))
        print('GMM Covariance by sklearn:\n', sk_cov)
        
        VC = VisualizationCluster(input_data)
        VC.plotLogLikelihood(likelihoods)
        print("\nVisualize my GMM implementation:")
        VC.plotClusteredData(clustering_result, np.asarray(mean).T)
        print("\nVisualize sklearn GMM implementation:")
        VC.plotClusteredData(sk_clustering_result, np.asarray(sk_mean).T)
    

"""Plot Input Data"""
VC = VisualizationCluster(input_data)
VC.plotInputData()

"""Fire K-means"""
KM = MyKMeans(input_data)
centroids = KM.main()


"""Fire GMM"""
GM = MyGMM(input_data, centroids)
GM.main()


    