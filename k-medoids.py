# -*- coding:utf-8 -*-
"""
@author: LIU
@file:k-medoids.py
@time:2016/11/1 16:25
"""
import numpy as np
import time
import random
import math


# ------------读数据文件----------------------------------------------------------------------------
def read_data():
    data_matrix = []
    data_label = []
    file_data = open('mnist.txt')
    for line in file_data.readlines():
        line.strip()
        temp = list(map(float, line.split(',')))
        data_label.append(temp[len(temp)-1])
        del temp[(len(temp)-1)]
        data_matrix.append(temp)
        file_data.close()
    return np.matrix(data_matrix), data_label


# ------------计算距离矩阵----------------------------------------------------------------------------
def weight_graph(sample):
    weight_gph = [[float("inf") for col in range(sample.shape[0])] for row in range(sample.shape[0])]
    for i in xrange(sample.shape[0]):
        temp = np.sum(np.abs(np.tile(sample[i], (sample.shape[0], 1)) - sample), axis=1)
        for j in xrange(temp.shape[0]):
            weight_gph[i][j] = float(temp[j][0])
    return weight_gph


# ------------初始化cluster represent ----------------------------------------------------------------
def init_medoids(sample, k):
    num, dim = sample.shape
    init_centroids = {}
    init_index = []
    for i in range(k):
        index = int(random.uniform(0, num))
        init_centroids[index] = sample[index, :]
        init_index.append(index)
    return init_centroids, init_index


# ----------total cost ------------------------------------------------------------------------
def total_dist(sample, centroids, distant_mat):
    num = sample.shape[0]
    total_dst = 0.0
    medoids = {}
    for idx in centroids:
        medoids[idx] = []
    for i in range(num):
        choice = None
        min_dst = float('inf')
        for m in medoids:
            tmp = distant_mat[m][i]
            if tmp < min_dst:
                choice = m
                min_dst = tmp
        medoids[choice].append(i)
        total_dst += min_dst
    return total_dst, medoids


# ----------k-medoids---------------------------------------------------------------------
def k_medoids(sample, centroids, index_arr, distant_mat):
    dist, k_medoids = total_dist(sample, centroids, distant_mat)
    cur_dist = float('inf')
    best_choice = []
    best_res = {}
    iter_count = 0
    while 1:
        for m in index_arr:
            for item in k_medoids[m]:
                if item != m:
                    idx = index_arr.index(m)
                    swap_temp = m
                    index_arr[idx] = item
                    tmp, medoids_ = total_dist(sample, index_arr, distant_mat)
                    if tmp < cur_dist:
                        best_choice = list(index_arr)
                        best_res = dict(medoids_)
                        cur_dist = tmp
                    index_arr[idx] = swap_temp
        iter_count += 1
        if best_choice == index_arr:
            break
        if cur_dist <= dist:
            dist = cur_dist
            k_medoids = best_res
            index_arr = best_choice
    print "iter   : ", iter_count
    return cur_dist, best_choice, best_res


# -------------计算k_NN,k取值为3\6\9---------------------------------------------------------
def k_NN(sample, k):
    k_nn = [[float(0) for col in range(sample.shape[0])] for row in range(sample.shape[0])]
    for s in xrange(sample.shape[0]):
        inputsample = sample[s]
        e_distance = []
        samples_num = sample.shape[0]
        nearest_array = np.tile(inputsample, (samples_num, 1)) - sample
        for i in range(0, nearest_array.shape[0]):
            square = np.matrix(nearest_array[i]) * np.matrix((nearest_array[i]).T)
            e_distance.append(math.sqrt(np.sum(square, axis=1)))
        dis_index = np.argsort(e_distance)
        distance_array = {}
        # calculate k nearest distance
        for i in xrange(k):
            distance_array[dis_index[i]] = e_distance[dis_index[i]]
            k_nn[s][dis_index[i]] = 1
            k_nn[dis_index[i]][s] = 1
            # the max voted class will return
    return k_nn


# ------------spectral clustering ------------------------------------------------------
def Spectral_Clustering(dist_matrix, cluster_num):

    num = len(dist_matrix)
    D = 1 / np.sqrt(np.sum(dist_matrix, axis=1))
    DN = np.diag(np.array(D))
    LapN = np.eye(num) - np.dot(np.dot(DN, dist_matrix), DN)
    U, s, V = np.linalg.svd(LapN)
    kerN = U[:, num - cluster_num + 1:num]
    for i in range(num):
        kerN[i, :] = kerN[i, :] / np.linalg.norm(kerN[i, :])
    del DN, U, s, V
    return kerN


# ------------计算purity---------------------------------------------------------------
def purity_gini(sample, medoids, index_arry, label_arr):
    p_ij = {}
    gini = {}
    purity = 0.0
    denomin = 0
    numer = 0
    for i in index_arry:
        max_type = 0
        label_n = {}
        var_sum = 0
        for label in label_arr:
            label_n[label] = 0
        for index in medoids[i]:
            label_n[label_arr[index]] += 1
        if (label_n[label_arr[index]]) > max_type:
            max_type = label_n[label_arr[index]]

        for key in label_n.keys():
            var_sum += pow(float(label_n[key]) / float(len(medoids[i])), 2)
        gini[len(medoids[i])] = 1 - var_sum
        p_ij[i] = (float(max_type) / float(len(medoids[i])))

    for i in index_arry:
        purity += float(len(medoids[i])) * p_ij[i]
    purity /= float(sample.shape[0])

    for key in gini.keys():
        numer += key * gini[key]
        denomin += key
    gini_coef = float(numer) / float(denomin)
    return purity, gini_coef


if __name__ == "__main__":
    sample_data, sample_label = read_data()
    method = raw_input("input method \n  0 : k_medoids  \n  1 : spectral \n  ")  # 输入 clustering method
    t1 = time.clock()

    # k_medoids 方法-----------------------------------------------------------------------------------------
    centroids, index_arr = init_medoids(sample_data, 10)
    if method == 0:
        # 计算距离矩阵----------------------------------------------------------------------------------------
        distance_matrix = weight_graph(sample_data)
        total_distance, index_arry, medoids_dict = k_medoids(sample_data, centroids, index_arr, distance_matrix)

    # spectralclustering方法---------------------------------------------------------------------------------
    else:
        # 计算k_NN矩阵----------------------------------------------------------------------------------------
        distance_matrix = k_NN(sample_data, 9)
        ker = Spectral_Clustering(distance_matrix, 10)
        total_distance, index_arry, medoids_dict = k_medoids(ker, centroids, index_arr, distance_matrix)

    # 计算purity , gini_index
    p, g = purity_gini(sample_data, medoids_dict, index_arry, sample_label)
    print "purity : ", p
    print "gini   : ", g
    t2 = time.clock()
    print "time   : ", t2-t1
